import bcrypt
import getpass


def get_password():
    """Ask the user for a password twice and ensure they match."""
    while True:
        password1 = getpass.getpass("Enter your password: ")
        password2 = getpass.getpass("Confirm your password: ")

        if password1 == password2:
            return password1  # Return the valid password
        else:
            print("Error: Passwords do not match. Please try again.\n")


# Get the password from the user
password = get_password()

# Generate the hashed password
hashed = bcrypt.hashpw(password.encode(), bcrypt.gensalt())

print("\nYour hashed password (store this in .env or secrets.toml):")
print(hashed.decode())
