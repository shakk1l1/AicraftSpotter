print("----------------------------------------------------------------")
print(r"   _____  .__                             _____  __   _________              __    __                ")
print(r"  /  _  \ |__|______   ________________ _/ ____\/  |_/   _____/_____   _____/  |__/  |_  ___________ ")
print(r" /  /_\  \|  \_  __ \_/ ___\_  __ \__  \\   __\\   __\_____  \\____ \ /  _ \   __\   __\/ __ \_  __ \ ")
print(r"/    |    \  ||  | \/\  \___|  | \// __ \|  |   |  | /        \  |_> >  <_> )  |  |  | \  ___/|  | \/")
print(r"\____|__  /__||__|    \___  >__|  (____  /__|   |__|/_______  /   __/ \____/|__|  |__|  \___  >__|")
print(r"        \/                \/           \/                   \/|__|                          \/      ")
print("----------------------------------------------------------------")
print(" ")
print("This is a command line interface for an aircraft spotter based on ia models\n"
      "You need to have a .txt file with the name of the image and the manufacturer/family\n")
print("The model can predict a test group with an accuracy score determined by the model\n"
        "It can also predict the family and manufacturer of a specific image\n"
        "It will save the model in a .pkl or .pth file\n")

## Check if the path file exists
try:
    # Check if the path file exists
    from path import path
    pathtest = path
except Exception as e:
    # If the path file does not exist, create it
    print("path file not found")
    print("Creating path file...")
    print("Please enter the path to the data folder (finish with .../AicraftSpotter/data)")
    new_path = input('path to data folder (finish with .../AicraftSpotter/data):')      # get the absolute path to the data folder

    # Create the path file wich is a python file with a variable called path
    with open('path.py', 'w') as f:
        f.write('path = ' + "r'" + new_path + "'")
    print("Path created\n")
    # Import the path variable from the path file
    from path import path