#!/bin/bash

# Some notes from Zachary Brown:
#
# Basic python zeromq stream reader:
# > python DEigerStream -i 10.139.1.5 -p 9999 -f junk.log
#
#Just in case it was not clear, I wanted to summarize getting the detector up and running again, and the sequence (and commands) to take an image after the power outage:
#
#Hardware:
#
#    Verify that the chiller is running and set to 22*C
#    Verify that the dry air is flowing (just enough that you can feel it on your lip should be plenty)
#    Verify that the blue LED on the power brick is lit
#    Verify that the detector's POWER switch is depressed
#    Verify that the DCU is powered on and booted, and that the iDRAC LED or display panel do not show any errors
#
# 
#Software:
#
#    Initialize the detector: 
#        curl -X PUT 10.139.1.5/detector/api/1.8.0/command/initialize
#    Enable FileWriter or Stream (or both): 
#        curl -X PUT 10.139.1.5/filewriter/api/1.8.0/config/mode -H "Content-Type:application/json" -d '{"value":"enabled"}'
#    Arm the detector: 
#        curl -X PUT 10.139.1.5/detector/api/1.8.0/command/arm
#    Trigger the detector:  
#         curl -X PUT 10.139.1.5/detector/api/1.8.0/command/trigger
#
#
#Options:
#
#    change count time (eyes open time): 
#        curl -X PUT 10.139.1.5/detector/api/1.8.0/config/count_time -H "Content-Type:application/json" -d '{"value":10}'
#    change framerate: 
#        curl -X PUT 10.139.1.5/detector/api/1.8.0/config/frame_time -H "Content-Type:application/json" -d '{"value":10}'
#    change the images per sequence:  
#         curl -X PUT 10.139.1.5/detector/api/1.8.0/config/nimages -H "Content-Type:application/json" -d '{"value":100}'
#
#Create Listener:
#
#	make zmq listener for header
# 	cd /home/labuser/Projects/Dectris/reborn/developer/rkirian/projects/cxls/dectris/fromzach/DEigerStream
#
#	conda activate reborn
#  	
#	python DEigerStream.py -i 10.139.1.5 -f junk.log
#

conda activate reborn
export PYTHONPATH=/home/labuser/Projects/Dectris/reborn



alias initialize="curl -X PUT 10.139.1.5/detector/api/1.8.0/command/initialize"
alias disarm="curl -X PUT 10.139.1.5/detector/api/1.8.0/command/disarm"
alias arm="curl -X PUT 10.139.1.5/detector/api/1.8.0/command/arm"
alias trigger="curl -X PUT 10.139.1.5/detector/api/1.8.0/command/trigger"

enable_monitor () {
curl -X PUT 10.139.1.5/monitor/api/1.8.0/config/mode -H "Content-Type:application/json" -d '{"value":"enabled"}'
}

enable_filewriter () {
curl -X PUT 10.139.1.5/filewriter/api/1.8.0/config/mode -H "Content-Type:application/json" -d '{"value":"enabled"}'
}

enable_stream () {
curl -X PUT 10.139.1.5/stream/api/1.8.0/config/mode -H "Content-Type:application/json" -d '{"value":"enabled"}'
}

# How many images to collect in the series 
nimages () {
curl -X PUT 10.139.1.5/detector/api/1.8.0/config/nimages -H "Content-Type:application/json" -d "{\"value\":${1}}"
}

ntrigger () {
curl -X PUT 10.139.1.5/detector/api/1.8.0/config/ntrigger -H "Content-Type:application/json" -d "{\"value\":${1}}"
}

# Time between readouts or inverse of collection rate
frame_time () {
curl -X PUT 10.139.1.5/detector/api/1.8.0/config/frame_time -H "Content-Type:application/json" -d "{\"value\":${1}}"
}

# Exposure time
count_time () {
curl -X PUT 10.139.1.5/detector/api/1.8.0/config/count_time -H "Content-Type:application/json" -d "{\"value\":${1}}"
}

# Options are:  ????
trigger_mode () {
curl -X PUT 10.139.1.5/detector/api/1.8.0/config/trigger_mode -H "Content-Type:application/json" -d "{\"value\":\"${1}\"}"
}


#Fore filewriter mode!
nimages_per_file () {
curl -X PUT 10.139.1.5/filewriter/api/1.8.0/config/nimages_per_file -H "Content-Type:application/json" -d "{\"value\":${1}}"
}


filter_nimages(){
    # name_pattern

    local destination_path="/home/labuser/Projects/Dectris/test/temp_data"                          #destination path of downloaded images.

    series=1
    iteration=1
    local max_iterations=100                                                                        # !! Assuming 100 is enough easily increased !!
    found_image=()                                                                                  # creating empty array

    while true; do                                                                                 # loop 1
            found_files=0                                                                           # Reset the count of found files for each series
            while true; do                                                                          # loop 2
                filename="series_${series}_data_00000$iteration.h5"                                 # definition of file of current itteration
                if find "$destination_path" -maxdepth 1 -name "$filename" -print -quit | grep -q .; then    
                                                                                                    # if found in the destination with name specified, print all found files, 
                                                                                                    # quits when the file is found, | redirects output, grep searches the specified pattern, -q quiet mode, 
                                                                                                    # '.' regular expression that matches any single character.
                    
                    echo "Found file: $filename" 
                    ((found_files++))                                                               # Increment count of found files
                else
                    break                                                                           # if not found then loop 2 terminated.
                fi
                ((iteration++))                                                                     # increment iteration for the next file

                if ((iteration > max_iterations)); then                                             # if iterations is greater than max, breaks loop 2
                    break
                fi
            done

                if ((found_files == 0)); then                                                       # if the array of files is eqivalent to 0 iterate series, set iteration to 1.
                        ((series++))
                        iteration=1
                        if ((series > 100)); then                                                   # if series > 100 break loop 1
                                break
                        fi      
                        continue
                fi
                ((series++))
                iteration=1   
        done
}

#filter_nimages but changed t filter master files.
filter_master_nimages(){
    local destination_path="/home/labuser/Projects/Dectris/test/temp_data"
    series=1    
    found_image=()                                                                                  

    while true; do                                                                                
            found_files=0                                                                           
            while true; do                                                                          
                filename="series_${series}_master.h5"                                                 
                if find "$destination_path" -maxdepth 1 -name "$filename" -print -quit | grep -q .; then    
                    echo "Found file: $filename" 
                    ((found_files++))                                                               
                    ((series++))
                else
                    break
                fi
   
            done

                if ((found_files == 0)); then
                        ((series++))
                        if ((series > 100)); then
                                break
                        fi      
                        continue
                fi
                ((series++))
        done
}

download_images_from_IP(){                                                                              # retrives all files on 10.139.1.5/data
    local image_url="http://10.139.1.5/data"                                                            # local host data url
    local destination_path="/home/labuser/Projects/Dectris/test/temp_data"                              # want to download in this location. 
    local output_dir="temp_data"                                                                            
    wget -r -nH --cut-dirs=1 --no-parent --no-check-certificate --reject "index.html*" -P "$destination_path" "$image_url"  
        # instructs to get files from url, and to download them recursively ignoring SSL cert checks.
        # -r is recursive, tells wget to follow links and download files from URL and any other resources. 
        # --cut-dirs=1 specifies the number of directories (1) to cut from the path when saving the files. 
        # --no-parent avoids downloading the files from directories higher in the three than initial URL
        # --no-check-certificate disables SSL cert checks, downloads anyways even if cannot be validated. 
        # --reject "index.html*" telling not to downloadany file starting with index.html
        # -P "$desintation_path" specifies the path where to be saved. 
        # -"$image_url" url of what to download
    echo -en '\n'                                                                                            
    filter_nimages                                                                                      #calling filter function to find all the files we want to display
    echo -en '\n'
}

#view image(s) and/or master files. 
albula_(){
    source /home/labuser/Documents/vscode/select_files.sh
}



#open adxv from Scripps Research Institute to display HDF5 images
adxv_(){
    #find adxv file in linux file system ask series of questions to determine which file to run
    file_paths=($(find /home/labuser/Downloads /home/labuser/Documents -name adxv.x86_64Debian10))      #variable of all paths found within /home/labuser stored as an array

    if [[ ${#file_paths[@]} -eq 0 ]]; then                                                              #if the array_paths is empty then message is displayed and exits the prompts
        echo "The 'adxv.x86_64Debian10' file was not found in the specified directories."
        return
    fi

    echo "The 'adxv.x86_64Debian10' file was found in the following locations:"                         
    for ((i = 0; i < ${#file_paths[@]}; i++)); do                                                       # for each file in the stored array, its then displayed. 
        echo "$(($i + 1)). ${file_paths[$i]}"
    done
    echo

    ask_choice ${#file_paths[@]}                                                                        # calls ask_choice function and prompts user for valid option number (arguement is the total length of array)
    chosen_path=${file_paths[$(($choice - 1))]}                                                         #from ask_choice function, chosen path is stored as the chosen_path variable

    if [[ -f "$chosen_path" ]]; then
        echo "Running 'adxv.x86_64Debian10' from chosen path: $chosen_path"                             #runs the script of the chosen_path and makes sure its executable.
        chmod +x "$chosen_path"
        "$chosen_path"

    else
        echo "Invalid path. The file 'adxv.x86_64Debian10' was not found in the chosen location"        # path is invalid and cannot be found. 
    fi

}
#asking which path to use (used adxv_ function)
ask_choice(){
    local max_option=$1                                                         #function ask_choice has one input argument called max_option 
    read -p "Please choose an option (1-$max_option): " choice                  #reads input and stores variable as choice, and displays range of options
    if ! [[ "$choice" =~ ^[1-$max_option]$ ]]; then                             # if argument checks if the input matches the pattern of a number between 1 and max_option
        echo "Invalid choice. Please enter a number between 1 and $max_option"  # if invalid -> message
        ask_choice $max_option                                                  # again recursively calls itself to get valid input.
    fi
}



alias get_count_time="curl -X GET 10.139.1.5/detector/api/1.8.0/config/count_time"
alias get_frame_time="curl -X GET 10.139.1.5/detector/api/1.8.0/config/frame_time"
alias get_nimages="curl -X GET 10.139.1.5/detector/api/1.8.0/config/nimages"


check_settings () {
    get_count_time
    get_frame_time
    get_nimages
}


status () {
    curl -X GET 10.139.1.5/detector/api/1.8.0/status/state/configure
}

# reboot DCU
# initialize


#alias change_value="curl -X PUT 10.139.1.5/detector/api/1.8.0/config/${param}" -H "Content-Type:application/json" -d '{"value":${value}"}'
#echo $change_value


#NOT encorperated in any function, just want to keep it since it could be helpful in the future.
#helpful function to evaluate yes/no cases (used adxv_ function)
yes_no(){
        read -p "$1 (y/n): " choice                         # reads input, input from user stored as choice variable 
        case "$choice" in                                   # choice variable has 2 cases true/yes or false/no
        y|Y) return 0;;
        n|N) reutrn 1;;
        *) echo "Invalid input. Please enter 'y' or 'n'."   #for all other potential options, invalid option
            yes_no "$1";;                                   #recursively calls itself to choose valid option
        esac
}

choose_option() {
    read -p "Choose an option (data/master): " selected_option
    case "$selected_option" in
    data|Data) echo "You chose: $selected_option";;
    master|Master) echo "You chose: $selected_option";;
    *) echo "Invalid option. Please choose either 'data' or 'master'."
       choose_option;;
    esac
}
