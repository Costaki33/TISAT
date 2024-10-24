## TexNet Injection-Seismic Analysis Tool
By RE. Constantinos Skevofilax and PhD. Alexandros Savvaidis of BEG TexNet

### Introduction 
The TexNet Injection-Seismic Analysis Tool (TISAT) is designed to efficiently analyze the relationship between oil and gas fracking operations and the occurrence of induced earthquakes.

The tool plots reported well injection volumes and pressures, and calculates the daily bottomhole pressure experienced underground for the N closest oil and gas wells to a specific earthquake. The goal is to assess the potential influence of these operations on earthquake activity.

Traditionally, in the field of seismology, it is recognized that oil and gas operations can induce earthquakes, as evidenced by the increased volume and frequency of seismic activity in oil and gas fields. However, the underlying mechanisms behind this connection are not yet fully understood. The goal of this tool is to provide an accessible way to visualize trends in these fields, helping to develop a clearer understanding of how human activities contribute to earthquake induction.

## Features
- Visualizes well injection volumes and pressures provided by either TexNet or B3.
- Calculates daily bottomhole pressures for nearby wells.
- Plots injection data trends, including missing and incomplete records.
- Generates reports and visual outputs for deeper analysis.

## Directory Structure

### 1. Storage of Generated Outputs
This directory will store all images and files generated by TISAT.

### 2. Data Source Directory
This directory will hold critical data resources that the tool will reference for processing calculations.

## How to Use 

1. **Clone the Repository:**
   Clone the repository to your local environment using:
   ```bash
   git clone https://github.com/Costaki33/TISAT.git
   ```
2. **Requirements**

   Please have the latest Python version installed.
   Similarly, review the `requirements.txt` file for the necessary libraries you need to have installed.
   You may alternatively run the following to install the required libraries on your machine: 
   ```bash
   pip install -r requirements.txt
   ```

4. **Setup Directories:**
   - Create two directories:
     - One to store generated outputs (images and files).
       Modify the following path in the `tisat.py` script:
        ```python
            # Line 24:
            OUTPUT_DIR = '/your/output/directory/path'
        ```
     - Another to hold data sources for processing (IE. Strawn Formation Data File)
     
5. **Gather Required Data:**
   - Gather the Strawn Formation Data file (Latitude, Longitude, and Z Depth) and put it into your data folder.
        Modify the following path in the `tisat.py` script:
        ```python
            # Line 23:
            STRAWN_FORMATION_DATA_FILE_PATH = '/your/earthquake_data/TopStrawn_RD_GCSWGS84.csv'
        ```
   - Gather API Tube OD/ID File and put it into your data folder.
        Modify the following path in the `friction_loss_calc.py` script:
        ```python
            # Line 14:
            tubing_information = '/your/earthquake_data/cmpl_tubing_data.csv'
        ```
## Code Logic

- **Input Earthquake Data:** The user provides earthquake data from the **SeisComP FDSNWS Event - URL Builder in CSV format**.

- **Find Closest Wells:** The tool locates the closest wells to the earthquake within a set range. Well data is fetched from the TexNet Injection Reporting Tool's API.

- **Data Quality Analysis:**
  - **Complete Data:** Well Injection BBL and Avg Pressure PSIG is available.
  - **Incomplete Data:** Well Injection BBL available but Avg Pressure PSIG is missing.
  - **Missing Data:** Both Well Injection BBL and Avg Pressure PSIG are missing.

- **Data Validation:** Validates injection data up to a provided year cutoff before the earthquake (IE. 5 years prior).

- **Calculate Bottomhole Pressure:** Calculates bottomhole pressure using the formula provided by Jim Moore (RRC):
    ```python
    Bottomhole pressure = surface pressure + hydrostatic pressure - deltaP
    ```
    - **Hydrostatic pressure formula:** `0.465 psi/ft X depth (ft)`
    - **Friction loss (DeltaP):** Computed using the Colebrook-White equation for turbulent flow.
   
   *Note regarding bottomhole pressure calculation, the equation above is the ideal formula to be used when calculating. However, due to either a lack of precise of well depth information available, when calculating
    deltaP (friction loss), the depth used may either be the **depth of the packer** or the **depth of the well**. Ideally it is the depth of the packer, but if not available the depth of the well is used.*
    *Please review the `friction_loss_calc.py` and `tisat.py` to get a better understanding of how we handle for lack of well depth information*
  
- **Well Sorting:** Wells are sorted as deep or shallow based on the Strawn Formation, with shallow wells plotted on the top of the figure and vice versa for deep wells.

- **Visualization:** Generates plots and histograms for injection pressure trends, differentiating between deep and shallow wells.

## Running the Code


Run the following in your code environment:
### Data Source: TexNet 
```bash
python3 tisat.py 1
```
It will prompt you to enter a directory path where the outputted files will go. If it doesn't exist, TISAT will automatically make the directory for you in the working space where TISAT is: 
```bash
Enter the output directory file path: 
```
It will then prompt you to input a TexNet event ID and fetch earthquake data from the URL Builder. The script will automatically open the URL Builder: 
```bash
Click on the following link to fetch earthquake data:
http://scdb.beg.utexas.edu/fdsnws/event/1/builder
Enter the earthquake data in CSV format: 
```
CSV file refers to the format of the earthquake. For example:
```bash
Enter the earthquake data in CSV format: texnet2024ophu,2024-07-26T14:28:29.143846Z,32.76580810546875,-100.65941787347559,3.2958984375,5.136142178709405,"Western Texas"

Information about the current earthquake:
{'Event ID': 'texnet2024ophu', 'Latitude': 32.76580810546875, 'Longitude': -100.65941787347559, 'Origin Date': '2024-07-26', 'Origin Time': '14:28:29', 'Local Magnitude': 5.14} 
```
If you do not have access to the TexNet URL Builder (you must be on the BEG VPN to access), there is a format that the URL Builder uses to generate the CSV format, which is as follows: 
```bash
EventID,Time,Latitude,Longitude,Depth/km,Magnitude,EventLocationName
```
You can go to the [TexNet Catalog](https://catalog.texnet.beg.utexas.edu/), search for an earthquake you are interested in, and recreate the format to input it into TISAT as shown below:

![Catalog Example](https://github.com/Costaki33/tisat/raw/main/images/catalog_example.png)

You then will be asked to input a search range in kilometers, which will allow TISAT to gather all well information within said radius, with the starting point being the earthquake epicenter:
```bash
Enter the range in kilometers (E.g. 20km): 
```
Finally, you will be asked to send a year cutoff for analysis leading up to the earthquake. Do not make the cutoff request longer than the total length of available prior information that is available to the TexNet Injection Reporting Tool, as there will be a lack of information the plots can reference. For example, the TexNet Injection Reporting Tool as information up until 2017. It is not recommended to do a year prior than that on this data source.  
```bash
Enter the year cutoff you would like to analyze prior to the earthquake: (E.g. 5 yrs): 
```

### Data Source: B3 
```bash
python3 tisat.py 0
```
After prompting for an output directory, TISAT will ask the following: 
```bash
Please provide B3 data filepath (In CSV format):
```
If you have access to a B3 subscription, please enter the MonthlyInjection data as a CSV file to be used as the data source. 

## Outputs
The script outputs:

- Plots showing injection pressures for shallow and deep wells.
- Histograms of complete, incomplete, and missing data for a given well over time.
- Text and CSV files containing cleaned and compiled well injection data used for plot generation.

Below are sample output plots of injection activity over time as well as the quality of the reported data within the vicinity of the 4.5M Scurry-Fisher, TX earthquake:

![Daily Injection Plot](https://github.com/Costaki33/TISAT/raw/main/images/daily_injection_plot_texnet2024ophu_range25.0km.png)

![Average Pressure Plot](https://github.com/Costaki33/TISAT/raw/main/images/event_texnet2024ophu_listed_avg_pressure_range25.0km.png)

![Well Data Histogram 15100308](https://github.com/Costaki33/TISAT/raw/main/images/well_data_histogram_15100308_range25.0km.png)

![Well Data Histogram 15133086](https://github.com/Costaki33/TISAT/raw/main/images/well_data_histogram_15133086_range25.0km.png)

## Applications
**TISAT** enables detailed analysis of injection activity across multiple wells at different depths, helping seismologists identify unusual injection patterns that may be linked to induced seismicity in a given region. 
TISAT has been instrumental in identifying hazardous well activity in the Scurry-Fisher region of Texas, which was linked to the inducement of four M4+ earthquakes in the area. 

## Questions? 
Please reach out to **constantinos.skevofilax@austin.utexas.edu** for any questions about usage, code development, or applications!
