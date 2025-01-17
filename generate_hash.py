import bcrypt

password = "xxxxx"  # original password
hashed = bcrypt.hashpw(password.encode(), bcrypt.gensalt())  # geberate hash

print(hashed.decode())  # copy this to the .env file
