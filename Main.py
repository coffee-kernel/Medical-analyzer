import gradio as gr
from PIL import Image
import os
import base64
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langgraph.prebuilt import create_react_agent
from langchain_core.messages import HumanMessage
from langchain.tools import tool
import requests
import json
from io import BytesIO
import re

load_dotenv()

llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", google_api_key=os.getenv("GEMINI_API_KEY"))

def encode_image_to_base64(image: Image) -> str:
    buffered = BytesIO()
    image.save(buffered, format="JPEG")
    return base64.b64encode(buffered.getvalue()).decode()

def extract_prescription_text(image: Image):
    img_b64 = encode_image_to_base64(image)
    
    prompt = ChatPromptTemplate.from_template(
        """You are a medical document expert. Analyze this base64-encoded prescription image. 
        Extract key details as valid JSON only—no extra text: 
        {{"patient_name": "str or null", "date": "str or null", "medications": [{{"name": "str", "dosage": "str", "frequency": "str"}}], "doctor_name": "str or null"}}. 
        Prioritize legibility for handwriting; infer if unclear but flag uncertainties."""
    )
    
    chain = prompt | llm

    response = chain.invoke({
        "input": f"Image data: data:image/jpeg;base64,{img_b64}"
    })
    
    content = response.content.strip()
    print(f"Raw LLM Response: {content[:500]}...")
    
    try:
        # IMPROVED: Use regex to find first valid JSON object (handles prefixes safely)
        json_match = re.search(r'\{.*\}', content, re.DOTALL)  # DOTALL for multi-line
        if json_match:
            json_str = json_match.group(0)
            extracted = json.loads(json_str)
        else:
            raise json.JSONDecodeError("No JSON found", content, 0)
    except json.JSONDecodeError as e:  # FIXED: Correct exception name (Decode, not Decoder)
        print(f"JSON Parse Error: {e} - Raw: {content[:200]}...")  # Log for debug
        extracted = {"error": f"Failed to parse: No valid JSON in response (check image clarity). Raw snippet: {content[:100]}"}
    return extracted

@tool(description="Fetch FDA side effects and price estimate for a medication.")
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
agent = create_react_agent(llm, tools)

def analyze_full(image: Image):
    extracted = extract_prescription_text(image)
    meds = extracted.get("medications", [])
    insights = []
    
    for med in meds:
        messages = [HumanMessage(content=f"Fetch info for {med['name']}")]
        result = agent.invoke({"messages": messages})
        
        last_msg = result["messages"][-1].content
        insights.append({"med": med['name'], "info": last_msg})
    
    return extracted, insights

def generate_report(extracted, insights):
    if insights is None:
        insights = []
    if extracted is None:
        extracted = {"error": "No data extracted."}
    
    md = f"# Prescription Analysis Report\n\n**Disclaimer:** This is AI-generated; consult a doctor.\n\n## Extracted Details\n{json.dumps(extracted, indent=2)}\n\n## Drug Insights\n"
    if not insights:
        md += "*No medications found or fetch failed.*\n\n"
    else:
        for insight in insights:
            md += f"### {insight['med']}\n{insight['info']}\n\n"
    md += "## Recommendations\n- Verify dosages. - Monitor for interactions."
    
    return md

with gr.Blocks(title="Medical Analyzer") as demo:
    gr.Markdown("# Medical Prescription Analyzer")
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