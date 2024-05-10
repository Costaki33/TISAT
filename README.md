## TexNet Earthquake Injection Correlation Tool 
By Constantinos Skevofilax and Professor Alexandros Savvaidis of BEG TexNet

### Introduction 
The purpose of the Earthquake Injection Correlation Tool is to quickly analyze the correlation between oil & gas fracking operations and the inducement of earthquakes. This tool plots the bottom-hole pressure experienced underground daily with the N closest oil & gas wells to a given earthquake, with the goal of understanding its influence on earthquake occurrence.

Traditionally in the seismology field, it is understood that oil & gas operations are inducing earthquakes, which can be seen with the high volume and frequency of earthquakes that occur in oil & gas fields daily. However, this correlation isn't fully understood yet, and the goal of this tool is to easily visualize the trends that are occurring in these fields to develop a better understanding of how humans induce earthquakes. 

This code is currently applied for the Midland Basin, Texas but could be applied to any location if formatted correctly. 

### How to Use 

1. Pull the repository on your given environment
2. Export from [TexNet](https://injection.texnet.beg.utexas.edu/) the latest injection well data and put it in your desired data folder 
3. Download the TexNet Earthquake Catalog from [TexNet, University of Texas at Austin](https://www.beg.utexas.edu/texnet-cisr/texnet/earthquake-catalog) and put it in your data folder
4. Provide a formation to calculate well type (shallow or deep)

### Code Logic

1. Takes a given earthquake, either user-provided or from the earthquake catalog (sorted from earliest to latest)
2. Finds the N closest wells (10) to said earthquake within an X KM range (20 is chosen from seismology research), using API number as the common identification number for a given well
3. Validates that the N closest wells have injection data that fall within +/- 1 year of earthquake occurrence
4. If validated, we calculate the bottom-hole pressure for a given well on a given day using the following formula Jim Moore of the RRC provided
```
Bottomhole pressure = surface pressure + hydrostatic pressure.
Simple formula for hydrostatic pressure is 0.465 psi/ft X depth (ft).
0.465 psi/ft is a typical value of the hydrostatic density of injection fluid.
It can vary by the actual density of the saltwater being injected.
```

This script requires an internet connection, as to get the well depth there is an API written that opens the RRC's well information web page and returns the listed well depth.

5. Sort the wells by type, deep or shallow, based on the Strawn Formation
6. Plot 

There are two ways to run this code: 

#### Method 1 
Method 1 will run through the entire TexNet Earthquake Catalog. Simply run the following command and the script will run automatically. 

```python
python3 inj_pandas.py 0
```

#### Method 2
Method 2 allows the user to choose any given earthquake and the script will plot independently.

```python
python3 inj_pandas.py 1
```
This will pop up a page, provided a given texnet event ID, make sure to click on the CSV formatting and copy/paste the earthquake information to the script. Be sure to be on the BEG TexNet VPN to access the page. 

### Output 
The script outputs a file called ```earthquake_info.txt```, which is in the following format: 

```python
TexNet Even ID, Latitude, Longitude, Date of Occurence, Time of Occurence, Magnitude, Distance between Previous Earthquake (KM), Time Lag Between Previous Earthquake and Current Earthquake
texnet2017nnah, 32.0883, -102.2565, 2017-07-12, 02:54:13, 2.0, 0.91, 22
texnet2017nnbc, 32.0856, -102.2511, 2017-07-12, 03:17:26, 2.0, 0.59, 0
texnet2017nnbm, 32.1149, -102.2725, 2017-07-12, 03:29:25, 2.1, 3.83, 0
```
