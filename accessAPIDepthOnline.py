from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

# Replace with the path to your chromedriver executable
service = Service('/usr/lib/chromium-browser/chromedriver')
driver = webdriver.Chrome(service=service)

# Open the website
driver.get('https://webapps2.rrc.texas.gov/EWA/wellboreQueryAction.do')

# Wait for the input fields to be present
first_input_field = WebDriverWait(driver, 10).until(
    EC.presence_of_element_located((By.NAME, 'searchArgs.apiNoPrefixArg'))
)
second_input_field = WebDriverWait(driver, 10).until(
    EC.presence_of_element_located((By.NAME, 'searchArgs.apiNoSuffixArg'))
)

# Split the API number
api_number = '31741196'
first_input = api_number[:3]  # '317'
second_input = api_number[3:]  # '41196'

# Fill the input fields
first_input_field.send_keys(first_input)
second_input_field.send_keys(second_input)

# Similarly, fill in other form fields using appropriate element identifiers

# Submit the form
submit_button = WebDriverWait(driver, 10).until(
    EC.element_to_be_clickable((By.XPATH, '//input[@type="submit"]'))
)
submit_button.click()

# Pause the execution here to prevent the page from closing
input("Please close the new page to resume execution.")