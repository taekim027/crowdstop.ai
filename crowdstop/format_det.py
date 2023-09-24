# Specify the input and output file paths
input_file_path = '../sompt22/train/SOMPT22-07/det/det.txt'
output_file_path = '../sompt22/train/SOMPT22-07/det/new_det.txt'

# Read the input file, remove spaces, and write to the output file
with open(input_file_path, 'r') as input_file, open(output_file_path, 'w') as output_file:
    for line in input_file:
        # Remove spaces using str.replace
        cleaned_line = line.replace(' ', '')
        # Write the cleaned line to the output file
        output_file.write(cleaned_line)

print(f"Spaces removed and saved to {output_file_path}")
