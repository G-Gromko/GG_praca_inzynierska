
import PySimpleGUI as sg
import os, io
from PIL import Image, ImageOps
from convert_to_MIDI import convert_files_to_MIDI

def main():
    file_list_column = [
        [ sg.Text("Choose files to convert from list:") ],
        [
            sg.Text("Image Folder"),
            sg.In(size=(30, 1), enable_events=True, key="-FOLDER-", readonly=True),
            sg.FolderBrowse(key="-FOLDER BR 1-"),
        ],
        [
            sg.Listbox(
                values=[], select_mode=sg.LISTBOX_SELECT_MODE_SINGLE, enable_events=True, size=(50, 40), key="-FILE LIST-"
            )
        ],
    ]

    file_list_column_ready = [
        [ sg.Text("Please select folder where MIDI files will be saved") ],
        [ 
            sg.In(size=(43,1), enable_events=True, key="-SAVE FOLDER-", readonly=True),
            sg.FolderBrowse(key="-FOLDER BR 2-"),
        ],
        [ sg.Text("------------------------------------------------------------------------------------------") ],
        [ sg.Text("Files to convert in order:") ],
        [
            sg.Listbox(
                values=[], select_mode=sg.LISTBOX_SELECT_MODE_SINGLE, enable_events=True, size=(50,37), key="-FILE LIST RDY-"
            )
        ]
    ]

    file_buttons = [
        [ sg.Button("", disabled=True, size=(0,10), button_color=("#64778d", "#64778d")) ],
        [ sg.Button(">>", key="-ADD-", size=(3,1)) ],
        [ sg.Button("", disabled=True, size=(0,33), button_color=("#64778d", "#64778d")) ],
    ]

    file_buttons_rdy= [
        [ sg.Button("", disabled=True, size=(0,8), button_color=("#64778d", "#64778d")) ],
        [ sg.Button("↑", key="-MV UP-", size=(3,1)) ],
        [ sg.Button("DEL", key="-REMOVE-", size=(3,1)) ],
        [ sg.Button("↓", key="-MV DOWN-", size=(3,1)) ],
        [ sg.Button("", disabled=True, size=(0,31), button_color=("#64778d", "#64778d")) ],
    ]

    # For now will only show the name of the file that was chosen
    image_viewer_column = [
        [ sg.Text(size=(45, 1), key="-TOUT-") ],
        [ sg.Image(size=(720, 720), key="-IMAGE-") ],
    ]

    convert_buttons = [
        [
            sg.Button("Convert", key="-CONVERT-", disabled=True), 
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

    def disable_UI(enable):
        window["-CONVERT-"].update(disabled=enable)
        # window["-ONE FILE-"].update(disabled=enable)
        window["-MV UP-"].update(disabled=enable)
        window["-REMOVE-"].update(disabled=enable)
        window["-MV DOWN-"].update(disabled=enable)
        window["-ADD-"].update(disabled=enable)
        window["-SAVE FOLDER-"].update(disabled=enable)
        window["-FOLDER-"].update(disabled=enable)
        window["-FOLDER BR 1-"].update(disabled=enable)
        window["-FOLDER BR 2-"].update(disabled=enable)


    def enable_conversion_buttons():
        if len(convert_files) == 0:
            window["-CONVERT-"].update(disabled=True)
        else:
            window["-CONVERT-"].update(disabled=False)


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
            save_path = ""
            if values["-SAVE FOLDER-"] == "":
                save_path = os.getcwd()
            else:
                save_path = values["-SAVE FOLDER-"]
                if not os.path.isdir(save_path):
                    save_path = os.getcwd()

            
            print("Converting files to MIDI ")
            disable_UI(True)
            sg.Popup(sg.popup_no_buttons("This may take a while.\nPlease be patient."))
            # mockup()
            convert_files_to_MIDI(convert_files, False, save_path)
            disable_UI(False)
            sg.Popup(sg.popup_no_buttons(f"Your MIDI files were saved in:\n{save_path}"))


    window.close()

if __name__ == "__main__":
    main()