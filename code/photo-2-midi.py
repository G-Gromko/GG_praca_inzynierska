
import PySimpleGUI as sg
import os, io
from PIL import Image, ImageOps
from convert_to_MIDI import convert_files_to_MIDI

def main():
    file_list_column = [
        [ sg.Text("Choose files to convert from list:") ],
        [
            sg.Text("Image Folder"),
            sg.In(size=(25, 1), enable_events=True, key="-FOLDER-"),
            sg.FolderBrowse(),
        ],
        [
            sg.Listbox(
                values=[], select_mode=sg.LISTBOX_SELECT_MODE_SINGLE, enable_events=True, size=(50, 40), key="-FILE LIST-"
            )
        ],
    ]

    file_list_column_ready = [
        [ sg.Text("Please select files with music score\n\nFiles to convert in order:") ],
        [
            sg.Listbox(
                values=[], select_mode=sg.LISTBOX_SELECT_MODE_SINGLE, enable_events=True, size=(50,40), key="-FILE LIST RDY-"
            )
        ]
    ]

    file_buttons = [
        [ sg.Button("", disabled=True, size=(0,10), button_color=("#64778d", "#64778d")) ],
        [ sg.Button(">>", key="-ADD-", size=(3,1)) ],
        [ sg.Button("", disabled=True, size=(0,41), button_color=("#64778d", "#64778d")) ],
    ]

    file_buttons_rdy= [
        [ sg.Button("", disabled=True, size=(0,8), button_color=("#64778d", "#64778d")) ],
        [ sg.Button("↑", key="-MV UP-", size=(3,1)) ],
        [ sg.Button("DEL", key="-REMOVE-", size=(3,1)) ],
        [ sg.Button("↓", key="-MV DOWN-", size=(3,1)) ],
        [ sg.Button("", disabled=True, size=(0,39), button_color=("#64778d", "#64778d")) ],
    ]

    # For now will only show the name of the file that was chosen
    image_viewer_column = [
        [ sg.Text(size=(72, 1), key="-TOUT-") ],
        [ sg.Image(size=(720, 800), key="-IMAGE-") ],
    ]

    convert_buttons = [
        [
            sg.Button("Convert", key="-CONVERT-", disabled=True), 
            sg.Checkbox("Combine to one file", key="-ONE FILE-", disabled=True), 
            sg.Checkbox("Also make PDF", key="-MAKE PDF-", disabled=True)
        ],
    ]

    # ----- Full layout -----
    layout = [
        [
            sg.vtop(sg.Column(file_list_column, element_justification='l')),
            sg.vtop(sg.Column(file_buttons, element_justification='c')),
            sg.vtop(sg.Column(file_list_column_ready, element_justification='l')),
            sg.VSeparator(),
            sg.vtop(sg.Column(file_buttons_rdy, element_justification='c')),
            sg.VSeparator(),
            sg.vtop(sg.Column(image_viewer_column)),
        ],
        [
            sg.Push(),
            sg.Column(convert_buttons, element_justification='r')
        ]
    ]

    window = sg.Window("Photo-2-MIDI", layout)


    def enable_conversion_buttons():
        if len(convert_files) == 0:
            window["-CONVERT-"].update(disabled=True)
            window["-ONE FILE-"].update(disabled=True)
            window["-MAKE PDF-"].update(disabled=True)
        else:
            window["-CONVERT-"].update(disabled=False)
            window["-ONE FILE-"].update(disabled=False)
            window["-MAKE PDF-"].update(disabled=False)


    def diplay_image(filename):
        if filename.endswith((".pdf", ".PDF")):
            image = Image.open("pdf_image.jpg")
        else:
            image = Image.open(filename)

        fx_image = ImageOps.exif_transpose(image)
        fx_image.thumbnail((720, 720))

        buffer = io.BytesIO()
        fx_image.save(buffer, format="PNG")

        window["-TOUT-"].update(filename)
        window["-IMAGE-"].update(data=buffer.getvalue())

    def move_image_up(filename, convert_files):
        idx = convert_files.index(filename)

        if idx > 0:
            convert_files.pop(idx)
            idx -= 1
            convert_files.insert(idx-1, filename)

        return convert_files, idx

    def move_image_down(filename, convert_files):
        idx = convert_files.index(filename)

        if idx < len(convert_files)-1:
            convert_files.pop(idx)
            idx += 1
            convert_files.insert(idx, filename)

        return convert_files, idx



    leng = 0
    idx = 0
    convert_files = []
    # Run the Event Loop
    while True:
        event, values = window.read()
        if event == "Exit" or event == sg.WIN_CLOSED:
            break
        # Folder name was filled in, make a list of files in the folder
        if event == "-FOLDER-":
            folder = values["-FOLDER-"]
            try:
                # Get list of files in folder
                file_list = os.listdir(folder)
            except:
                file_list = []

            fnames = [
                f
                for f in file_list
                if os.path.isfile(os.path.join(folder, f))
                and f.endswith((".png", ".png", ".PNG", ".JPG", ".pdf", ".PDF"))
                and f not in convert_files
            ]
            window["-FILE LIST-"].update(fnames)

        elif event == "-FILE LIST-":  # A file was chosen from the listbox
            try:
                filename = os.path.join(
                    values["-FOLDER-"], values["-FILE LIST-"][0]
                    ).replace("\\", "/")
                diplay_image(filename)
                enable_conversion_buttons()
            except:
                pass

        elif event == "-FILE LIST RDY-":
            try:
                filename = values["-FILE LIST RDY-"][0]
                diplay_image(filename)
                enable_conversion_buttons()
            except:
                pass

        elif event == "-ADD-":
            try:
                filename = os.path.join(
                    values["-FOLDER-"], values["-FILE LIST-"][0]
                    ).replace("\\", "/")
                
                if filename not in convert_files:
                    convert_files.append(filename)
                    window["-FILE LIST RDY-"].update(convert_files)

                enable_conversion_buttons()

            except:
                pass
        
        elif event == "-REMOVE-":
            try:
                filename = values["-FILE LIST RDY-"][0]
                
                convert_files.remove(filename)
                window["-FILE LIST RDY-"].update(convert_files)

                enable_conversion_buttons()
            except:
                pass


        elif event == "-MV UP-":
            try:
                filename = values["-FILE LIST RDY-"][0]
                cf, new_idx = move_image_up(filename, convert_files)

                convert_files = cf
                window["-FILE LIST RDY-"].update(convert_files)
                window["-FILE LIST RDY-"].update(set_to_index=[new_idx], scroll_to_index=new_idx)
            except:
                pass

        elif event == "-MV DOWN-":
            try:
                filename = values["-FILE LIST RDY-"][0]
                cf, new_idx = move_image_down(filename, convert_files)

                convert_files = cf
                window["-FILE LIST RDY-"].update(convert_files)
                window["-FILE LIST RDY-"].update(set_to_index=[new_idx], scroll_to_index=new_idx)
            except:
                pass

        elif event == "-CONVERT-":
            if values["-MAKE PDF-"] == True and values["-ONE FILE-"] == True:
                print("Converting to one MIDI file and making pdf")
                # convert_files_to_MIDI(convert_files, True, True)
            if values["-MAKE PDF-"] == False and values["-ONE FILE-"] == True:
                print("Converting to one MIDI file")
                # convert_files_to_MIDI(convert_files, False, True)
            if values["-MAKE PDF-"] == True and values["-ONE FILE-"] == False:
                print("Converting to multiple MIDI files and making multiple PDF")
                # convert_files_to_MIDI(convert_files, True, False)
            if values["-MAKE PDF-"] == False and values["-ONE FILE-"] == False:
                print("Converting to multiple MIDI files")
                # convert_files_to_MIDI(convert_files, False, False)


    window.close()

if __name__ == "__main__":
    main()