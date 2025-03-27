import math

def gcd(a, b):
    """Calculate the Greatest Common Divisor of a and b."""
    while b:
        a, b = b, a % b
    return a

def mod_inverse(a, m):
    """Calculate the modular multiplicative inverse of a modulo m."""
    # Ensure that a and m are coprime
    if gcd(a, m) != 1:
        raise ValueError(f"The value of 'a' ({a}) is not coprime with {m}. Choose another value.")
    
    # Using Extended Euclidean Algorithm to find modular inverse
    for i in range(1, m):
        if (a * i) % m == 1:
            return i
    return None

def affine_encrypt(plain_text, a, b):
    """
    Encrypt the plain text using the Affine cipher.
    
    E(x) = (ax + b) mod 26
    
    Parameters:
    plain_text (str): The text to encrypt.
    a (int): The multiplicative key (must be coprime with 26).
    b (int): The additive key.
    
    Returns:
    str: The encrypted text.
    """
    # Check if 'a' is coprime with 26
    if gcd(a, 26) != 1:
        raise ValueError(f"The value of 'a' ({a}) is not coprime with 26. Choose another value.")
    
    encrypted_text = ""
    
    for char in plain_text:
        if char.isalpha():
            # Convert to uppercase for consistency
            char = char.upper()
            
            # Convert letter to number (A=0, B=1, ..., Z=25)
            x = ord(char) - ord('A')
            
            # Apply affine transformation: E(x) = (ax + b) mod 26
            encrypted_num = (a * x + b) % 26
            
            # Convert back to letter
            encrypted_char = chr(encrypted_num + ord('A'))
            encrypted_text += encrypted_char
        else:
            # Keep non-alphabetic characters as they are
            encrypted_text += char
    
    return encrypted_text

def affine_decrypt(cipher_text, a, b):
    """
    Decrypt the cipher text using the Affine cipher.
    
    D(y) = a^(-1) * (y - b) mod 26
    
    Parameters:
    cipher_text (str): The text to decrypt.
    a (int): The multiplicative key (must be coprime with 26).
    b (int): The additive key.
    
    Returns:
    str: The decrypted text.
    """
    # Find the modular multiplicative inverse of 'a'
    a_inverse = mod_inverse(a, 26)
    
    decrypted_text = ""
    
    for char in cipher_text:
        if char.isalpha():
            # Convert to uppercase for consistency
            char = char.upper()
            
            # Convert letter to number (A=0, B=1, ..., Z=25)
            y = ord(char) - ord('A')
            
            # Apply affine decryption: D(y) = a^(-1) * (y - b) mod 26
            decrypted_num = (a_inverse * (y - b)) % 26
            
            # Convert back to letter
            decrypted_char = chr(decrypted_num + ord('A'))
            decrypted_text += decrypted_char
        else:
            # Keep non-alphabetic characters as they are
            decrypted_text += char
    
    return decrypted_text

def display_mapping(a, b):
    """Display the complete character mapping for the given key (a, b)."""
    print("\nCharacter Mapping:")
    print("Original: ", end="")
    for i in range(26):
        print(chr(i + ord('A')), end=" ")
    
    print("\nEncrypted:", end="")
    for i in range(26):
        encrypted = (a * i + b) % 26
        print(chr(encrypted + ord('A')), end=" ")
    print("\n")

# User input version
if __name__ == "__main__":
    print("Welcome to the Affine Cipher Program!")
    print("Common values for 'a' that are coprime with 26: 1, 3, 5, 7, 9, 11, 15, 17, 19, 21, 23, 25")
    
    # Get valid key parameters from the user
    while True:
        try:
            a = int(input("Enter the value for 'a' (must be coprime with 26): "))
            if gcd(a, 26) != 1:
                print(f"Error: {a} is not coprime with 26. Please choose another value.")
                continue
            
            b = int(input("Enter the value for 'b' (any integer): "))
            break
        except ValueError:
            print("Please enter valid integers for 'a' and 'b'.")
    
    # Display the key parameters
    print(f"\nAffine Cipher with parameters a={a}, b={b}")
    
    # Show character mapping
    display_mapping(a, b)
    
    # Menu for operations
    while True:
        print("\nChoose an option:")
        print("1. Encrypt a message")
        print("2. Decrypt a message")
        print("3. Exit")
        
        choice = input("Enter your choice (1-3): ")
        
        if choice == '1':
            plain_text = input("Enter the message to encrypt: ")
            try:
                cipher_text = affine_encrypt(plain_text, a, b)
                print(f"Encrypted message: {cipher_text}")
            except ValueError as e:
                print(f"Error: {e}")
                
        elif choice == '2':
            cipher_text = input("Enter the message to decrypt: ")
            try:
                decrypted_text = affine_decrypt(cipher_text, a, b)
                print(f"Decrypted message: {decrypted_text}")
            except ValueError as e:
                print(f"Error: {e}")
                
        elif choice == '3':
            print("Thank you for using the Affine Cipher Program!")
            break
            
        else:
            print("Invalid choice. Please enter 1, 2, or 3.")