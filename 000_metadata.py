#!/usr/bin/env python3

"""
Script: PrairieView & VoltageRecording XML Metadata Extraction + QC
Author: Boris Bouazza-Arostegui
Date: May 2025

Workflow:
- Load PrairieView XML file
- Parse scanner mode, laser power/wavelength, PMT gain/offset, zoom, objective, scan size
- Load VoltageRecording XML file
- Load VoltageOutput XML file
- QC comparison between imaging and ephys metadata
- Save summary metadata + QC report
- Generate signal plots for each channel
"""

import os
import xml.etree.ElementTree as ET
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# ==== FILE PATH SETUP ====
tif_path = r'your_path_here\example_Cycle00001.tif'  # Replace with your actual TIFF file path
tif_name = os.path.basename(tif_path)
tif_base = tif_name.split('_Cycle')[0]
directory = os.path.dirname(tif_path)
xml_path = os.path.join(directory, f'{tif_base}.xml')
volt_xml_path = os.path.join(directory, f'{tif_base}_Cycle00001_VoltageRecording_001.xml')
volt_output_path = os.path.join(directory, f'{tif_base}_Cycle00001_VoltageOutput_001.xml')

output_folder = directory
os.makedirs(output_folder, exist_ok=True)

def parse_pv_metadata(xml_file):
    tree = ET.parse(xml_file)
    root = tree.getroot()

    frame_period = pix_x = pix_y = None
    scan_mode = resonant_freq = None
    laser_power, laser_wavelength = {}, {}
    pmt_gain, pmt_offset = {}, {}
    zoom = objective = None
    microns_per_pixel_x = microns_per_pixel_y = None

    for elem in root.iter('PVStateValue'):
        key = elem.attrib.get('key')
        if key == 'framePeriod': frame_period = float(elem.attrib.get('value'))
        elif key == 'pixelsPerLine': pix_x = int(elem.attrib.get('value'))
        elif key == 'linesPerFrame': pix_y = int(elem.attrib.get('value'))
        elif key in ('scanMode', 'activeMode'): scan_mode = elem.attrib.get('value')
        elif key == 'resonantFrequency': resonant_freq = float(elem.attrib.get('value'))
        elif key == 'opticalZoom': zoom = float(elem.attrib.get('value'))
        elif key == 'objectiveMagnification': objective = elem.attrib.get('value')
        elif key == 'micronsPerPixelX': microns_per_pixel_x = float(elem.attrib.get('value'))
        elif key == 'micronsPerPixelY': microns_per_pixel_y = float(elem.attrib.get('value'))

        elif key == 'laserPower':
            for iv in elem.findall('IndexedValue'):
                laser_power[int(iv.attrib['index'])] = float(iv.attrib['value'])
        elif key == 'laserWavelength':
            for iv in elem.findall('IndexedValue'):
                laser_wavelength[int(iv.attrib['index'])] = float(iv.attrib['value'])
        elif key == 'pmtGain':
            for iv in elem.findall('IndexedValue'):
                pmt_gain[int(iv.attrib['index'])] = float(iv.attrib['value'])
        elif key == 'pmtOffset':
            for iv in elem.findall('IndexedValue'):
                pmt_offset[int(iv.attrib['index'])] = float(iv.attrib['value'])

    if None in (frame_period, pix_x, pix_y):
        raise ValueError("Missing essential XML metadata.")

    frame_rate = round(1.0 / frame_period, 4)
    frame_count = len(root.findall('.//Frame'))
    total_time_sec = round(frame_count * frame_period, 2)

    scan_size_x = scan_size_y = None
    if microns_per_pixel_x and microns_per_pixel_y:
        scan_size_x = round(pix_x * microns_per_pixel_x, 2)
        scan_size_y = round(pix_y * microns_per_pixel_y, 2)

    return {
        "frame_rate": frame_rate,
        "pix_x": pix_x,
        "pix_y": pix_y,
        "frame_count": frame_count,
        "total_time_sec": total_time_sec,
        "scan_mode": scan_mode,
        "resonant_freq": resonant_freq,
        "laser_power": laser_power,
        "laser_wavelength": laser_wavelength,
        "pmt_gain": pmt_gain,
        "pmt_offset": pmt_offset,
        "zoom": zoom,
        "objective": objective,
        "scan_size_x": scan_size_x,
        "scan_size_y": scan_size_y
    }

def parse_voltage_recording_metadata(xml_file):
    tree = ET.parse(xml_file)
    root = tree.getroot()
    experiment = root.find('Experiment')

    rate = int(experiment.find('Rate').text)
    acq_time_ms = int(experiment.find('AcquisitionTime').text)

    signals = experiment.find('SignalList')
    signal_lines = []
    for signal in signals.findall('VRecSignal'):
        name = signal.find('Name').text
        unit = signal.find('Unit/UnitName').text
        multiplier = float(signal.find('Unit/Multiplier').text)
        divisor = float(signal.find('Unit/Divisor').text)
        gain = float(signal.find('Gain').text)
        scale = multiplier / divisor
        channel = signal.find('Channel').text
        enabled = signal.find('Enabled').text
        signal_lines.append(f"{name} (Ch {channel}) [{unit}], Enabled={enabled}, Gain={gain}, Scale={scale:.6f}")

    return rate, acq_time_ms, signal_lines

def parse_voltage_output_metadata(xml_file):
    tree = ET.parse(xml_file)
    root = tree.getroot()
    waveforms = root.findall('Waveform')
    lines = []
    for w in waveforms:
        name = w.findtext('Name')
        enabled = w.findtext('Enabled') == 'true'
        line = w.findtext('AOLine')
        units = w.findtext('Units')
        component = w.find('WaveformComponent_PulseTrain')
        desc = f"AO{line} '{name}' [{units}] (enabled={enabled})"
        if component is not None:
            desc += f": {component.findtext('Name')}, pulses={component.findtext('PulseCount')}, width={component.findtext('PulseWidth')} ms"
        lines.append(desc)
    return lines

def check_qc(imaging_time, acq_time, frame_rate, sr):
    errors = []
    if abs(imaging_time - (acq_time / 1000)) > 0.2:
        errors.append(f"⛔ Duration mismatch: imaging = {imaging_time}s, ephys = {acq_time / 1000:.2f}s")
    if sr < 500:
        errors.append(f"⚠️  Ephys sampling rate too low: {sr} Hz")
    if frame_rate < 100:
        errors.append(f"⚠️  Imaging frame rate seems low: {frame_rate} Hz")
    return errors

# ==== RUN PARSERS ====
pv = parse_pv_metadata(xml_path)
sr, acq_time, signal_lines = parse_voltage_recording_metadata(volt_xml_path)
waveform_lines = parse_voltage_output_metadata(volt_output_path)
qc_notes = check_qc(pv["total_time_sec"], acq_time, pv["frame_rate"], sr)

# ==== SAVE METADATA SUMMARY ====
with open(os.path.join(output_folder, "extracted_metadata.txt"), "w", encoding="utf-8") as f:
    f.write("--- PrairieView Scan Metadata ---\n")
    f.write(f"Frame rate: {pv['frame_rate']} Hz\n")
    f.write(f"Dimensions: {pv['pix_x']} x {pv['pix_y']} pixels\n")
    f.write(f"Total frames: {pv['frame_count']}\n")
    f.write(f"Total imaging time: {pv['total_time_sec']} s\n")
    f.write(f"Zoom: {pv['zoom']}\n")
    f.write(f"Objective: {pv['objective']}\n")
    if pv['scan_size_x'] and pv['scan_size_y']:
        f.write(f"Scan field size: {pv['scan_size_x']} µm x {pv['scan_size_y']} µm\n")

    f.write("\n--- Scanner Configuration ---\n")
    f.write(f"Scan mode: {pv['scan_mode'] or 'Unknown'}\n")
    if pv['resonant_freq']:
        f.write(f"Resonant frequency: {pv['resonant_freq']} Hz\n")

    f.write("\n--- Laser Configuration ---\n")
    for ch in sorted(pv["laser_power"]):
        wl = pv["laser_wavelength"].get(ch, 'N/A')
        f.write(f"Laser {ch}: {pv['laser_power'][ch]}% @ {wl} nm\n")

    f.write("\n--- PMT Settings ---\n")
    for ch in sorted(pv["pmt_gain"]):
        offset = pv["pmt_offset"].get(ch, 'N/A')
        f.write(f"PMT {ch}: Gain = {pv['pmt_gain'][ch]}, Offset = {offset}\n")

    f.write("\n--- Voltage Recording Metadata ---\n")
    f.write(f"Sampling rate: {sr} Hz\n")
    f.write(f"Acquisition duration: {acq_time / 1000:.2f} s\n")
    for line in signal_lines:
        f.write(f" - {line}\n")

    f.write("\n--- Voltage Output Metadata ---\n")
    for line in waveform_lines:
        f.write(f" - {line}\n")

    f.write("\n--- Quality Control ---\n")
    if qc_notes:
        for note in qc_notes:
            f.write(f"{note}\n")
    else:
        f.write("✅ All checks passed.\n")

print("✅ Metadata summary and QC saved.")

# ==== PLOT SIGNALS IF CSV EXISTS ====
csv_file = volt_xml_path.replace('_VoltageRecording_001.xml', '_VoltageRecording_001.csv')
if os.path.exists(csv_file):
    print(f"📊 Generating plots from: {csv_file}")
    df = pd.read_csv(csv_file)
    df.columns = df.columns.str.strip()

    time = df["Time(ms)"].values / 1000 if "Time(ms)" in df.columns else None
    primary = df["Primary"].values * 100 if "Primary" in df.columns else None
    secondary = df["Secondary"].values * 1000 if "Secondary" in df.columns else None
    ecg = df["ECG"].values if "ECG" in df.columns else None
    airpuff = df["AIRPUFF"].values if "AIRPUFF" in df.columns else None

    sns.set_style("darkgrid")
    sns.set_palette("colorblind")

    signals = [(primary, "Vm (mV)", "blue"), (secondary, "pA", "green"), (ecg, "ECG", "red"), (airpuff, "Airpuff", "black")]
    valid_signals = [(sig, label, color) for sig, label, color in signals if sig is not None and time is not None]

    fig, axes = plt.subplots(len(valid_signals), 1, figsize=(12, 2.5 * len(valid_signals)), sharex=True)
    if len(valid_signals) == 1:
        axes = [axes]  # ensure it's iterable

    for ax, (data, label, color) in zip(axes, valid_signals):
        sns.lineplot(x=time, y=data, ax=ax, color=color)
        ax.set_ylabel(label)
        ax.set_title(label if label != "Airpuff" else "Multi-Channel Signals")
    axes[-1].set_xlabel("Time (s)")
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, "Multi_Channel_Signals.png"), dpi=300)
    plt.close()

    channel_data = {
        "Primary (Vm mV)": (primary, "blue", "Vm (mV)"),
        "Secondary (pA)": (secondary, "green", "Current Injection (pA)"),
        "ECG": (ecg, "red", "ECG Signal"),
        "AIRPUFF": (airpuff, "black", "Airpuff Stimulus"),
    }

    for label, (data, color, ylabel) in channel_data.items():
        if data is not None and time is not None:
            fig, ax = plt.subplots(figsize=(12, 3))
            sns.lineplot(x=time, y=data, ax=ax, color=color)
            ax.set_title(label)
            ax.set_ylabel(ylabel)
            ax.set_xlabel("Time (s)")
            plt.tight_layout()
            filename = f"{label.replace(' ', '_').replace('(', '').replace(')', '').replace('/', '_')}.png"
            plt.savefig(os.path.join(output_folder, filename), dpi=300)
            plt.close()
            print(f"✅ Saved: {filename}")
else:
    print(f"⚠️ CSV data file not found for plotting: {csv_file}")