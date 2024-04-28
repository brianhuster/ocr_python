import platform

def check_os():
    os_name = platform.system()
    if os_name == 'Windows':
        print('You are using Windows')
    elif os_name == 'Linux':
        print('You are using Linux')
    elif os_name == 'Darwin':
        print('You are using MacOS')
    else:
        print('Unknown operating system')

check_os()