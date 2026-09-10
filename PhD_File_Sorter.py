import os
import shutil
import PyPDF2

# CHANGE DIRECTORIES BELOW TO YOUR LIKING - THIS WILL NOT WORK OUT OF THE BOX! ---------------------

# THIS PROGRAMME NEEDS TO BE RUN PERIODICALLY AS IT'LL NOT WORK IN THE BACKGROUND
# IF YOU WANT CONSTANT MONITORING - CONSIDER ADDING A WHILE LOOP AND TIME.SLEEP

pdf_folder = 'C:/Users/USER/PycharmProjects/PythonProject/PDF_Tool/PDF Downloads' 
# THE FOLDER WHERE PDFS GET DOWNLOADED - THIS CAN BE UPDATED IN GOOGLE SETTINGS

unsorted_pdf_loc = 'C:/Users/USER/Documents/Files/Paper PDFs/Unsorted'
# IF YOU HAVE AN UNSORTED FILES FOLDER FOR GENERAL PDFS

other_files_loc = 'C:/Users/USER/Downloads'
# WHERE TO SEND FILES IF THEY ARE NOT .PDF , .DOC , ....

suffix = ('.pdf', '.doc', '.docx', '.txt')
# UPDATE TO FILTER OUT ANY UNWANTED/WANTED FILE EXTENSIONS

for file in os.listdir(pdf_folder):
    if file.endswith(suffix):
        print(file)

        if file.endswith('.pdf'):
            with open(pdf_folder + '/' + file, 'rb') as pdf_file:
                reader = PyPDF2.PdfReader(pdf_file)

                metadata = reader.metadata
                print(f"Author: {metadata.author}")
                print(f"Title: {metadata.title}")
                print(f"Creation Date: {metadata.creation_date}")

            new_name = pdf_folder + '/' + metadata.title + '.pdf'
            os.rename(pdf_folder + '/' + file, new_name)
            move = shutil.move(new_name, unsorted_pdf_loc)

    else:
        # this moves any other files to the actual download folder
        print(f'File to be removed: {file}')
        move = shutil.move(pdf_folder + '/' + file, other_files_loc)
