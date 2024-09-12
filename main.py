import os
import re

import markdown
import pymupdf
from googlemaps import Client as GoogleMaps
from langchain.tools import BaseTool
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent
from markupsafe import Markup

llm = ChatOpenAI(model="gpt-4o-mini")


gmaps = GoogleMaps(os.getenv("GPLACES_API_KEY"))


PREFIX = """

You are a tool that leverages OpenAI's API, to extract and summarize key information from an annual report issued by a Homeowners Association (HoA).

Context:
HoAs regularly produce annual reports summarizing their activities, financial status, and future plans. These documents are crucial for members, stakeholders, and potential homebuyers but can often be lengthy and complex. Your purpose is to simplify this information when prompted by the user, making it more accessible.

"""


class PDFExtractorTool(BaseTool):
    name = "pdf_extractor"
    description = "Extracts text content from a provided PDF file."

    def _run(self, pdf_path):
        try:
            doc = pymupdf.open(pdf_path)
            text = ""
            for page_num in range(doc.page_count):
                page = doc.load_page(page_num)
                text += page.get_text()
            doc.close()
            return text
        except Exception as e:
            return f"An error occurred while extracting the PDF: {str(e)}"


system_message = SystemMessage(content=PREFIX)


pdf_extractor_tool = PDFExtractorTool(
    description="Extract text content from a PDF file given its path."
)
tools = [pdf_extractor_tool]


agent_executor = create_react_agent(llm, tools, messages_modifier=system_message)


def extract_and_remove_html(text):
    # Pattern to match HTML code block
    html_pattern = r"```html\s*([\s\S]*?)\s*```"

    # First look for any python code
    python_pattern = (
        r'<pre\s+class="codehilite"><code\s+class="language-python">(.*?)</code></pre>'
    )
    md_pattern = r"```python(.*?)```"
    python_match = re.search(python_pattern, text, re.DOTALL | re.IGNORECASE)
    md_match = re.search(md_pattern, text, re.DOTALL)
    code_match = python_match or md_match
    if code_match:
        print(text)
        code = code_match.group(1)
        code = code.replace("&quot;", '"')
        code = code.replace("&amp;", "&")
        code = code.replace("&lt;", "<")
        code = code.replace("&gt;", ">")
        code = code.replace("&#39;", "'")
        return None, "PDF Generated!", code

    # Search for the pattern in the text
    match = re.search(html_pattern, text, re.IGNORECASE)

    if match:
        # Extract the HTML code
        html_code = match.group(1).strip()

        # Remove the HTML code block from the original text
        text_without_html = re.sub(html_pattern, "", text, flags=re.IGNORECASE).strip()

        # Return both the extracted HTML and the text without HTML
        return Markup(html_code), text_without_html, False
    else:
        # If no HTML is found, return None for HTML and the original text
        return None, text, False


def process_markdown(text):
    """Convert Text to HTML and wrap in Markup"""
    html = markdown.markdown(text, extensions=["extra", "codehilite"])
    # Wrap the result in Markup to prevent auto-escaping
    return Markup(html)


def process_question(prompted_question, conversation_history, pdf_path):
    pdf_content = pdf_extractor_tool.run(pdf_path)

    context = "\n".join(
        [
            f"Q: {entry['question']}\nA: {entry['answer']}"
            for entry in conversation_history
        ]
    )
    consolidated_prompt = f"""
    PDF content:
    {pdf_content}

    Previous conversation:
    {context}

    New question: {prompted_question}

    Please answer the new question based on the PDF content and the context from the previous conversation if relevant.
    """
    prompt = consolidated_prompt if conversation_history else prompted_question

    content = []
    for s in agent_executor.stream({"messages": [HumanMessage(content=prompt)]}):
        for msg in s.get("agent", {}).get("messages", []):
            print(msg.content)
            content.append(process_markdown(msg.content))
        print("----")

    return content
