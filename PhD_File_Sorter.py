import os
import shutil
import PyPDF2
import time

# CHANGE DIRECTORIES BELOW TO YOUR LIKING - THIS WILL NOT WORK OUT OF THE BOX! ---------------------

# CHOOSE 'TRUE' IN THE FUNCTION TO TURN ON MANUAL MODE - THIS ALLOWS THE USER TO MANUALLY NAME THE FILE BEFORE IT GETS SENT
# LEAVING IT AS 'FASLE', IT'LL SEND THE FILE WITH THE ORIGINAL DOWNLOAD NAME

# THIS PROGRAMME NEEDS TO BE RUN PERIODICALLY AS IT'LL NOT WORK IN THE BACKGROUND
# IF YOU WANT CONSTANT MONITORING - UNHASH THE BOTTOM PIECE, YOU CAN ALTER THE FREQUENCY IN THE 'TIME.SLEEP( time (s) )' NOTE: MAY SLOW DOWN COMPUTER!

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
                    name = pdf_folder + '/' + file

                    reader = PyPDF2.PdfReader(pdf_file)

                    metadata = reader.metadata
                    print('-'*100)
                    print(f"Author: {metadata.author}")
                    print(f"Title: {metadata.title}")
                    print(f"Creation Date: {metadata.creation_date}")
                    print('-' * 100)

                try:
                    new_name = pdf_folder + '/' + metadata.title + '.pdf'
                    os.rename(name, new_name)
                    move = shutil.move(new_name, unsorted_pdf_loc)
                except Exception as e:
                    print('ERROR:', e)

                    if manual_mode == True:
                        temp_name = input(f'Enter a name manually for the file (include the file extension in the name!): {file}:\n')
                        updated_name = pdf_folder + '/' + temp_name

                        os.rename(name, updated_name)

                        print(f'Download will be sent to {unsorted_pdf_loc} with the name {temp_name}')
                        move = shutil.move(updated_name, unsorted_pdf_loc)

                    else:
                        print(f'File has been moved to {unsorted_pdf_loc} with the name {name}!\nEnsure to change it manually later!')
                        move = shutil.move(name, unsorted_pdf_loc)

        else:
            # this moves any other files to the actual download folder
            print(f'File to be removed: {file}')
            move = shutil.move(pdf_folder + '/' + file, other_files_loc)


clean = clean_folder(manual_mode=False)

# while True:
#     time.sleep(10)
#     clean = clean_folder(manual_mode=False)
#     print('Cleaned folder...')
