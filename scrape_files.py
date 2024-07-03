import os
import argparse

def scrape_files(directory, extensions, prefix, output_file):
    with open(output_file, 'w', encoding='utf-8') as outfile:
        for root, dirs, files in os.walk(directory):
            for file in files:
                if file.endswith(extensions) and file.startswith(prefix):
                    file_path = os.path.join(root, file)
                    with open(file_path, 'r', encoding='utf-8') as infile:
                        outfile.write(f"// File: {file_path}\n")
                        outfile.write(infile.read())
                        outfile.write("\n\n")

def main():
    parser = argparse.ArgumentParser(description='Scrape Python files into a single text file.')
    parser.add_argument('-d', '--directory', type=str, required=True,
                        help='Directory path to start scraping from')
    parser.add_argument('-e', '--extensions', type=str, nargs='+', default=['.py'],
                        help='Acceptable file extensions (default: .py)')
    parser.add_argument('-p', '--prefix', type=str, default='',
                        help='Prefix to match file names (default: None)')
    parser.add_argument('-o', '--output', type=str, default='output.txt',
                        help='Output file name (default: output.txt)')

    args = parser.parse_args()

    directory = args.directory
    extensions = tuple(args.extensions)
    output_file = args.output
    prefix = args.prefix

    scrape_files(directory, extensions, prefix, output_file)
    print(f"Python files scraped and saved to {output_file}")

if __name__ == '__main__':
    main()