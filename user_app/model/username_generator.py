import ctypes
import os
import sys

from cryptography.fernet import Fernet


class Credentials:

    def __init__(self):
        self.__username = ""
        self.__key = ""
        self.__password = ""
        self.__key_file = 'key.key'
        self.__time_of_exp = -1

    @property
    def password(self):
        return self.__password

    @password.setter
    def password(self, password):
        self.__key = Fernet.generate_key()
        f = Fernet(self.__key)
        self.__password = f.encrypt(password.encode()).decode()
        del f

    def create_cred(self):
        cred_filename = 'CredFile.ini'

        with open(cred_filename, 'w') as file_in:
            file_in.write("#Credential file:\nPassword={}\nExpiry={}\n"
                          .format(self.__password, self.__time_of_exp))
            file_in.write("++" * 20)

        if os.path.exists(self.__key_file):
            os.remove(self.__key_file)

        try:

            os_type = sys.platform
            if os_type == 'linux':
                self.__key_file = '.' + self.__key_file

            with open(self.__key_file, 'w') as key_in:
                key_in.write(self.__key.decode())
                if os_type == 'win32':
                    ctypes.windll.kernel32.SetFileAttributesW(self.__key_file, 2)
                else:
                    pass

        except PermissionError:
            os.remove(self.__key_file)
            print("A Permission error occurred.\n Please re run the script")
            sys.exit()

        self.__username = ""
        self.__password = ""
        self.__key = ""
        self.__key_file = ""


def create_username(username):
    creds = Credentials()
    creds.username = username
    creds.password = username
    creds.expiry_time = '-1'
    creds.create_cred()
