import gradio as gr
from PIL import Image
import os
import base64
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import ChatPromptTemplate
from langchain.tools import tool
from langchain.agents import creat_react_agent, AgentExecutor
import requests
import json
from io import BytesIO

load_dotenv()

llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=os.getenv("GEMINI_API_KEY"))

def encode_image_to_base64(image: Image) -> str:
    buffered = BytesIO()
    image.save(buffered, format="JPEG")
    return base64.b64encode(buffered.getvalue()).decode()

def extract_prescription_text(image: Image):
    img_b64 = encode_image_to_base64(image)
    
    prompt = ChatPromptTemplate.from_template(
        """Analyze this base64-encoded prescription image. Extract as JSON: 
        {{"patient_name": "...", "date": "...", "medications": [{{"name": "...", "dosage": "...", "frequency": "..."}}], "doctor_name": "..."}}. 
        Handle handwriting if present. Be accurate for medical terms."""
    )
    
    chain = prompt | llm

    response = chain.invoke({
        "input": f"Image data: data:image/jpeg;base64,{img_b64}",
        "img_path": "N/A"
    })
    
    try:
        extracted = json.loads(response.content)
    except:
        extracted = {"error": "Failed to parse response"}
    return extracted

@tool
def fetch_drug_info(med_name: str) -> str:
    url = f"https://api.fda.gov/drug/label.json?search=openfda.brand_name:\"{med_name}\"&limit=1"
    if os.getenv("FDA_API_KEY"):
        url += f"&api_key={os.getenv('FDA_API_KEY')}"
    resp = requests.get(url)
    data = resp.json()
    side_effects = data['results'][0].get('adverse_reactions', 'N/A') if data['results'] else 'N/A'
    price = "Check GoodRx"
    return f"Side effects: {side_effects[:200]}... \nEstimated Price: {price}"

tools = [fetch_drug_info]
agent = creat_react_agent(llm, tools)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

def analyze_full(image: Image):
    extracted = extract_prescription_text(image)
    meds = extracted.get("medications", [])
    insights = []
    
    for med in meds:
        result = agent_executor.invoke({"input": f"Fetch info for {med['name']}"})
        insights.append({"med": med['name'], "info": result['output']})
    return {"extracted": extracted, "insights": insights}

def generate_report(extracted, insights):
    md = f"# Prescription Analysis Report\n\n**Disclaimer:** This is AI-generated; consult a doctor.\n\n## Extracted Details\n{json.dumps(extracted, indent=2)}\n\n## Drug Insights\n"
    for insight in insights:
        md += f"### {insight['med']}\n{insight['info']}\n\n"
    md += "## Recommendations\n- Verify dosages. - Monitor for interactions."
    return md

with gr.Blocks() as demo:
    gr.Markdown("# Medical Prescription Analyzer (Free Gemini Edition)")
    img_input = gr.Image(type="pil")
    extract_btn = gr.Button("Analyze")
    extracted_out = gr.JSON(label="Extracted Data")
    insights_out = gr.JSON(label="Insights")
    report_out = gr.Markdown(label="Full Report")
    extract_btn.click(analyze_full, inputs=img_input, outputs=[extracted_out, insights_out]).then(
        generate_report, inputs=[extracted_out, insights_out], outputs=report_out
    )

if __name__ == "__main__":
    demo.launch()