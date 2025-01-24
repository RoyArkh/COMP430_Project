import csv
import sys

def replace_first_column(doc1, doc2, output_doc):
    # Read the first column of the first document
    with open(doc1, 'r') as file1:
        reader1 = csv.reader(file1)
        next(reader1)  # Skip header
        first_column_doc1 = [row[0] for i, row in enumerate(reader1) if i < 700]

    # Read the second document and replace its first column
    with open(doc2, 'r') as file2:
        reader2 = csv.reader(file2)
        header_doc2 = next(reader2)  # Get header
        rows_doc2 = [row for i, row in enumerate(reader2) if i < 700 and i != 99]  #skip the 100th row ->bcs single word

    # Replace the first column of the second document
    for i, row in enumerate(rows_doc2):
        row[0] = first_column_doc1[i]

    # Write the updated content to the output document
    with open(output_doc, 'w', newline='') as output_file:
        writer = csv.writer(output_file)
        writer.writerow(header_doc2)  # Write header
        writer.writerows(rows_doc2)  # Write updated rows

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python script.py <nbhd> <others> <output_doc>")
        sys.exit(1)

    input_doc1 = sys.argv[1]
    input_doc2 = sys.argv[2]
    output_doc = sys.argv[3]

    replace_first_column(input_doc1, input_doc2, output_doc)

