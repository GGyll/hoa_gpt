import os

import PyPDF2
from langchain.chains import LLMChain
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from unidecode import unidecode


def clean_text(text):
    """Clean the extracted text."""
    text = unidecode(text)
    text = text.replace("\x00", "")
    text = " ".join(text.split())
    return text


def extract_text_from_pdf(pdf_path):
    """Extract text from a PDF file, returning a list of pages."""
    with open(pdf_path, "rb") as file:
        reader = PyPDF2.PdfReader(file)
        pages = []
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                pages.append(clean_text(page_text))
            else:
                print(f"Warning: Could not extract text from a page in {pdf_path}")
    return pages


# Initialize the language model
llm = OpenAI(temperature=0.5)


BASE_TEMPLATE = """
Summarize the key information about the Homeowners Association and its board from page {page_num} of the annual report.

IMPORTANT: If there is info indicating that the numbers are in thousands or millions, please convert them to actual numbers.


\n\n{page_content}\n\nSummary:

"""
# Create prompt templates
summary_prompt = PromptTemplate(
    input_variables=["page_num", "page_content"],
    template=BASE_TEMPLATE,
)

LOAN_TEMPLATE = """
Extract and list any information about loans of the association from page {page_num} of the annual report if and only if you find any loan information on that page.

Instructions:
1. Look for key terms such as financial institutions, loan amounts, interest rates, and loan terms (and their equivalents in Swedish).
2. If you find any loan information, provide a concise summary of the loans found.
3. If you do not find any loan-related information, you must return an empty string without any explanation.
4. If loan info is found, include an explanation of why you believe it is loan info

IMPORTANT: 
- Do not say "No loan information found" or any variant of this.
- Do not apologize or explain the absence of loan information.
- If you do not find loan information, your response must include the following subtring "5&NOTFOUND"
- "Fastighetsinteckning" is not loan-related

Page content:
\n\n{page_content}


"""
loans_prompt = PromptTemplate(
    input_variables=["page_num", "page_content"],
    template=LOAN_TEMPLATE,
)

COMPARISON_TEMPLATE = """
return a JSON object with a 2D array of the loan information. The first row should be the headers and the subsequent rows should be the loan information.

Here is the header row ["Loan institution", "Loan Amount"]

Match the loan info to the respective headers and return the information in the 2D array format.

If you do not find any loan-related info in the provided text, just return None, do not try to make anything up, it is very important that the information you provide is actually in the supplied text.

IMPORTANT: Do not try to provide any example input or output, just provide the requested information if found.


"""
comparison_prompt = PromptTemplate(
    input_variables=["loan_info"],
    template=COMPARISON_TEMPLATE,
)
# Create LLMChains
summary_chain = LLMChain(llm=llm, prompt=summary_prompt)
loans_chain = LLMChain(llm=llm, prompt=loans_prompt)
comparison_chain = LLMChain(llm=llm, prompt=comparison_prompt)


def analyze_report(pdf_path):
    """Analyze the HOA annual report."""
    # Extract text from PDF
    pages = extract_text_from_pdf(pdf_path)

    # Process each page
    summaries = []
    loan_info = []
    for i, page_content in enumerate(pages, start=1):
        summaries.append(summary_chain.run(page_num=i, page_content=page_content))
        _loan_info = loans_chain.run(page_num=i, page_content=page_content)
        if "5&NOTFOUND" not in _loan_info:
            loan_info.append(_loan_info)

    # Combine the results
    full_summary = "\n\n".join(
        f"Page {i+1} Summary:\n{summary}" for i, summary in enumerate(summaries)
    )
    full_loan_info = "\n\n".join(
        f"Page {i+1} Loans:\n{loans}"
        for i, loans in enumerate(loan_info)
        if loans.strip()
    )

    return pages, full_summary, full_loan_info


def main():
    print("Welcome to the HOA Annual Report Analyzer!")
    pdf_path = input("Please enter the path to the HOA annual report PDF: ")

    if not os.path.exists(pdf_path):
        print("Error: The specified file does not exist.")
        return

    print("\nAnalyzing the report...")
    pages, summary, loans = analyze_report(pdf_path)

    print("\nExtracted text (first 500 characters of first page):")
    print(pages[0][:500] if pages else "No text extracted")

    print("\nSummary of the HOA and its board:")
    print(summary)

    print("\nLoan information:")
    print(loans)
    print(comparison_chain.run(loan_info=loans))


if __name__ == "__main__":
    main()
