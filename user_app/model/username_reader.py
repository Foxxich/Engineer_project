from cryptography.fernet import Fernet


def read_username():
    cred_filename = 'CredFile.ini'
    with open('key.key', 'r') as key_in:
        key = key_in.read().encode()

    f = Fernet(key)
    with open(cred_filename, 'r') as cred_in:
        lines = cred_in.readlines()
        config = {}
        for line in lines:
            tuples = line.rstrip('\n').split('=', 1)
            if tuples[0] in 'Password':
                config[tuples[0]] = tuples[1]

        passwd = f.decrypt(config['Password'].encode()).decode()
        return passwd
