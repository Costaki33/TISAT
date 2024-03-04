from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException
from statistics import mean


def get_api_depth(api_number):
    service = Service('/usr/lib/chromium-browser/chromedriver')
    driver = webdriver.Chrome(service=service)
    api_number = str(api_number)
    try:
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
        first_input = api_number[:3]
        second_input = api_number[3:]

        # Fill the input fields
        first_input_field.send_keys(first_input)
        second_input_field.send_keys(second_input)

        # Similarly, fill in other form fields using appropriate element identifiers

        # Submit the form
        submit_button = WebDriverWait(driver, 10).until(
            EC.element_to_be_clickable((By.XPATH, '//input[@type="submit"]'))
        )
        submit_button.click()

        # Wait for the new page to load after submitting the form
        wait = WebDriverWait(driver, 10)
        api_depth_element = wait.until(EC.presence_of_element_located((By.XPATH,
                                                                       '/html/body/table[2]/tbody/tr/td/form/table/tbody/tr[2]/td/table/tbody/tr[2]/td/table/tbody/tr[3]/td[10]')))

        # Get the text of the API Depth element
        api_depth = api_depth_element.text

        return api_depth
    except TimeoutException:
        print(f"Timeout error: No results found for API {api_number}. Using mean well depth instead.")
        return None
    finally:
        driver.quit()  # Close the browser window


def calculate_mean_api_depth(closest_wells_api_nums):
    valid_api_depths = []
    for api in closest_wells_api_nums:
        api_depth = get_api_depth(api)
        if api_depth is not None:
            # Convert api_depth to float and append to valid_api_depths
            try:
                valid_api_depths.append(float(api_depth))
            except ValueError:
                print(f"Warning: API Depth {api_depth} for API {api} is not a valid float. Skipping.")

        if not valid_api_depths:
            print("No valid API depths found. Exiting.")
            return None

        mean_api_depth = mean(valid_api_depths)
        return mean_api_depth
