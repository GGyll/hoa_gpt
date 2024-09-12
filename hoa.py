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

# Create prompt templates
summary_prompt = PromptTemplate(
    input_variables=["page_num", "page_content"],
    template="Summarize the key information about the Homeowners Association and its board from page {page_num} of the annual report:\n\n{page_content}\n\nSummary:",
)

LOAN_TEMPLATE = """
Extract and list any information about loans of the association from page {page_num} of the annual report if and only if you find any loan information on that page:\n\n{page_content}:
Look for key terms such as financial institutions, loan amounts, interest rates, and loan terms (and their equivalents in Swedish). If you find any loan information, please provide a summary of the loans found, otherwise, move to the next page.



"""
loans_prompt = PromptTemplate(
    input_variables=["page_num", "page_content"],
    template=LOAN_TEMPLATE,
)

# Create LLMChains
summary_chain = LLMChain(llm=llm, prompt=summary_prompt)
loans_chain = LLMChain(llm=llm, prompt=loans_prompt)


def analyze_report(pdf_path):
    """Analyze the HOA annual report."""
    # Extract text from PDF
    pages = extract_text_from_pdf(pdf_path)

    # Process each page
    summaries = []
    loan_info = []
    for i, page_content in enumerate(pages, start=1):
        summaries.append(summary_chain.run(page_num=i, page_content=page_content))
        loan_info.append(loans_chain.run(page_num=i, page_content=page_content))

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


if __name__ == "__main__":
    main()
