## TexNet Earthquake Injection Analysis Tool 
By Constantinos Skevofilax and Professor Alexandros Savvaidis of BEG TexNet

### Introduction 
The purpose of the Earthquake Injection Correlation Tool is to quickly analyze the connection between oil & gas fracking operations and the inducement of earthquakes. This tool plots the bottom-hole pressure experienced underground daily with the N closest oil & gas wells to a given earthquake, with the goal of understanding its influence on earthquake occurrence.

Traditionally in the seismology field, it is understood that oil & gas operations are inducing earthquakes, which can be seen with the high volume and frequency of earthquakes that occur in oil & gas fields. However, this connection isn't fully understood yet, and the goal of this tool is to easily visualize the trends that are occurring in these fields to develop a better understanding of how humans induce earthquakes. 

This code is currently applied for the Midland Basin, Texas but could be applied to any location if formatted correctly. 

### How to Use 

1. Pull the repository on your given environment
2. Create a directory to store the images and files that are created
   ```
   Replace OUTPUT_DIR = '/home/skevofilaxc/Documents/earthquake_plots' with your file path 
   ```
3. Gather Strawn Formation Latitude, Longitude and Z depth file and store in the earthquake_data folder
   ```
   Replace STRAWN_FORMATION_DATA_FILE_PATH = '/home/skevofilaxc/Documents/earthquake_data/TopStrawn_RD_GCSWGS84.csv' with your file path 
   ```
3. Gather API Tube OD ID File and store in the earthquake_data folder
   ```
   Replace tubing_information = '/home/skevofilaxc/Documents/earthquake_data/2024-03-06T08.47.26.270812_2024-03-05 cmpl tubing.py.csv' with your file path 
   ```
4. Have access to the BEG VPN to access the SeisComP FDSNWS Event - URL Builder to access the URL Builder 

### Code Logic

1. User provides earthquake information from the SeisComP FDSNWS Event - URL Builder in CSV formation
2. Finds the closest wells to said earthquake within an X KM range (20 is chosen from seismology research), using API number as the common identification number for a given well from the [TexNet Injection Tool API](https://injection.texnet.beg.utexas.edu/apidocs)
3. Create a histogram for a given well of the complete, incomplete and missing injection data. 
    - Complete: Well Injection BBL and Avg Pressure PSIG is available 
    - Incomplete: Well Injection BBL is available but Avg Pressure PSIG is not
    - Missing: Both Well Injection BBL and Avg Pressure PSIG are not available 
4. Validates that the closest wells have injection data that fall within +/- 1 year of earthquake occurrence
5. If validated, we calculate the bottom-hole pressure for a given well on a given day using the following formula Jim Moore of the RRC provided
```
Bottomhole pressure = surface pressure + hydrostatic pressure - deltaP.

Quote from Jim Moore of the RRC: 
Simple formula for hydrostatic pressure is 0.465 psi/ft X depth (ft).
0.465 psi/ft is a typical value of the hydrostatic density of injection fluid.
It can vary by the actual density of the saltwater being injected.

deltaP is the flowwing tubing friction loss within the pipe and it is calculated as so: 
1. The fluid is considered to be Newtonian Fluid 
2. Given an API number, the outer diameter (OD) and inner diameter (ID) is found using a list defined from the Halliburton Red Book Engineering Tables
3. Units (defined from the Petroleum Production Systems Textbook by Michael J. Economides, A. Daniel Hill, Christine Ehlig-Economides, Ding Zhu)
    roh = 64.3 lbm/ft^3 = 1.03-specific gravity of water 
    1.48 = constant to represent the general averages expressed in EQN. 7-7 in the PPS Book 
    q = injected BBL/day
    D = inner diameter (in) 
    mu = 0.6 = viscosity of water at bottomhole conditions
    
    Nre = (1.48*q*roh) / D * mu 
    
4. Laminar vs Turbulent Flow
    From Scott Rosenquist of the RRC: "[Assume] 0.0001 as an estimate [for pipe roughness], which is a fairly smooth state"
    if Nre < 2100: 
        # Laminar Flow
        friction factor = 16 / Nre 
    else: 
        # Turbulent Flow 
        # Use the Colebrook White Equation as advised by Scott 
        friction factor = Colebrook White equation from function created by [robm] (https://pypi.org/project/colebrook/)
        
5. Calculating DeltaP
    fluid_velocity_ft_s = (4 * injected_bbl * 5.615 * (1 / 86400)) / (math.pi * (inner_diameter_inches / 12)**2)  # ft/s
    deltaP = ((2 * friction_factor * 64.3 * (fluid_velocity_ft_s**2) * packer_depth_ft) / (32.17 * (inner_diameter_inches / 12))) * (1/144) # PSI
```

This script requires an internet connection, as to get the well depth there is an API written that opens the RRC's well information web page and returns the listed well depth.

6. Sort the wells by type, deep or shallow, based on the Strawn Formation
7. Plot 


#### Running Code 
Run the following in your code environment 

```python
python3 inj_pandas_optimized.py 0
```
This will pop up a page, provide a texnet event ID. Make sure to click on the CSV formatting and copy/paste the earthquake information to the script. Be sure to be on the BEG TexNet VPN to access the page. 
```python
python3 inj_pandas_optimized.py 0
Click on the following link to fetch earthquake data:
http://scdb.beg.utexas.edu/fdsnws/event/1/builder
Enter the earthquake data in CSV format:
```
Paste a CSV format of a given earthquake such as: 

```python
Enter the earthquake data in CSV format: texnet2020ochr,2020-07-19T11:35:39.489071Z,31.93725586,-102.3432736,6.749511719,2.511918765,"Western Texas"
```
Enter a range you would like to explore

```python
Enter the range in kilometers (E.g. 20km): 10
```
### Output 
The script outputs several files, including the shallow and deep plot of the injection pressure for a given earthquake, histograms of the complete, missing, and incomplete data 
data for a given well over time, and .txt files of the injection data for the shallow and deep plot for the user to recreate the plot on their own with the spliced data. 

