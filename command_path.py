def Commandpath():
    """
    Handle path-related commands for setting and displaying the folder path.
    """
    c_p = input('/path >> ')
    match c_p:
        case 'help':
            print("List of commands")
            print('     esc: escape')
            print('     show: show existing path')
            print('     new: create new path')
            Commandpath()
        case "esc":
            return None
        case "show":
            try:
                f = open("command_path.py", "r")
            except FileNotFoundError:
                print("No path found, please create a new one")
                Commandpath()
                return
            print(f.read())
            f.close()
            Commandpath()
        case "new":
            new_path = input('new path:')
            with open('command_path.py', 'w') as f:
                f.write('path = ' + "'" + new_path + "'")
            print("Path saved!")
            Commandpath()
        case _:
            print("unknown command")
            print("use help for commands list")
            Commandpath()

    return None