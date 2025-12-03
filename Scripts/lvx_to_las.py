import struct
import numpy as np
import laspy
import csv
import argparse
import os

def process_package(data, offset):
    """Process one package from a .LVX file starting at offset. 
    Works for Single First, Double and Triple return .lvx files using Cartesian Coordinate Systems (and Both Rep and Non-rep scan modes).
    Returns:
    - a list of points as tuples (x, y, z, intensity, return_number)
    - an IMU record as (timestamp, gyro_x, gyro_y, gyro_z, acc_x, acc_y, acc_z) if present, else None.
    - the new offset after reading the package."""
    #Package header
    device_index = data[offset]
    offset += 1
    version = data[offset]
    offset += 1
    slot_id = data[offset]
    offset += 1
    lidar_id = data[offset]
    offset += 1
    reserved = data[offset]
    offset += 1
    status_code = struct.unpack('<I', data[offset:offset+4])[0]
    offset += 4
    timestamp_type = data[offset]
    offset += 1
    data_type = data[offset]
    offset += 1
    timestamp = struct.unpack('<Q', data[offset:offset+8])[0]
    offset += 8

    points = []
    imu_record = None

    if data_type == 2:
        # Cartesian Single Return: 96 points, 14 bytes total.
        for i in range(96):
            x = int.from_bytes(data[offset:offset+4], 'little', signed=True)
            y = int.from_bytes(data[offset+4:offset+8], 'little', signed=True)
            z = int.from_bytes(data[offset+8:offset+12], 'little', signed=True)
            reflectivity = data[offset+12]
            tag = data[offset+13]
            offset += 14
            if not (x == 0 and y == 0 and z == 0 and reflectivity == 0):
                points.append((x, y, z, reflectivity, 1))
    elif data_type == 4:
        # Cartesian Double Return: 48 points, each with two returns, 28 bytes total.
        for i in range(48):
            x1 = int.from_bytes(data[offset:offset+4], 'little', signed=True)
            y1 = int.from_bytes(data[offset+4:offset+8], 'little', signed=True)
            z1 = int.from_bytes(data[offset+8:offset+12], 'little', signed=True)
            refl1 = data[offset+12]
            tag1 = data[offset+13]
            x2 = int.from_bytes(data[offset+14:offset+18], 'little', signed=True)
            y2 = int.from_bytes(data[offset+18:offset+22], 'little', signed=True)
            z2 = int.from_bytes(data[offset+22:offset+26], 'little', signed=True)
            refl2 = data[offset+26]
            tag2 = data[offset+27]
            offset += 28
            if not (x1 == 0 and y1 == 0 and z1 == 0 and refl1 == 0):
                points.append((x1, y1, z1, refl1, 1))
            if not (x2 == 0 and y2 == 0 and z2 == 0 and refl2 == 0):
                points.append((x2, y2, z2, refl2, 2))
    elif data_type == 7:
        # Cartesian Triple Return: 30 points, each with three returns, 42 bytes total.
        for i in range(30):
            x1 = int.from_bytes(data[offset:offset+4], 'little', signed=True)
            y1 = int.from_bytes(data[offset+4:offset+8], 'little', signed=True)
            z1 = int.from_bytes(data[offset+8:offset+12], 'little', signed=True)
            refl1 = data[offset+12]
            tag1 = data[offset+13]
            x2 = int.from_bytes(data[offset+14:offset+18], 'little', signed=True)
            y2 = int.from_bytes(data[offset+18:offset+22], 'little', signed=True)
            z2 = int.from_bytes(data[offset+22:offset+26], 'little', signed=True)
            refl2 = data[offset+26]
            tag2 = data[offset+27]
            x3 = int.from_bytes(data[offset+28:offset+32], 'little', signed=True)
            y3 = int.from_bytes(data[offset+32:offset+36], 'little', signed=True)
            z3 = int.from_bytes(data[offset+36:offset+40], 'little', signed=True)
            refl3 = data[offset+40]
            tag3 = data[offset+41]
            offset += 42
            if not (x1 == 0 and y1 == 0 and z1 == 0 and refl1 == 0):
                points.append((x1, y1, z1, refl1, 1))
            if not (x2 == 0 and y2 == 0 and z2 == 0 and refl2 == 0):
                points.append((x2, y2, z2, refl2, 2))
            if not (x3 == 0 and y3 == 0 and z3 == 0 and refl3 == 0):
                points.append((x3, y3, z3, refl3, 3))
    elif data_type == 6:
        # IMU data Cartesian: 24 bytes; unpack 6 floats.
        imu = struct.unpack('<6f', data[offset:offset+24])
        offset += 24
        imu_record = (timestamp, imu[0], imu[1], imu[2], imu[3], imu[4], imu[5])
    else:
        offset += 24

    return points, imu_record, offset

def process_frame(data, offset, frame_end):
    """Process one frame starting at offset until frame_end.
    Returns:
    - points from the frame,
    - IMU records from the frame,
    - new offset after processing the frame."""
    frame_points = []
    frame_imu = []
    while offset < frame_end:
        pts, imu, offset = process_package(data, offset)
        if pts:
            frame_points.extend(pts)
        if imu:
            frame_imu.append(imu)
    return frame_points, frame_imu, offset

def create_las_from_points(points, las_file):
    """Creates a .las file and converts mm to m based on scale factors. (.lvx files use mm)"""
    if not points:
        print(f"No points to write for {las_file}.")
        return
    arr = np.array(points)
    header = laspy.LasHeader(point_format=3, version="1.2")
    header.x_scale = 0.001 # Scale: converting mm to meters
    header.y_scale = 0.001
    header.z_scale = 0.001
    las = laspy.LasData(header)
    las.x = (arr[:, 0] * 0.001).astype(float)
    las.y = (arr[:, 1] * 0.001).astype(float)
    las.z = (arr[:, 2] * 0.001).astype(float)
    las.intensity = arr[:, 3].astype(np.uint16)
    las.return_number = arr[:, 4].astype(np.uint8)
    las.write(las_file)
    print(f"LAS file written: {las_file} with {len(points)} points.")

def write_imu_csv(imu_data, csv_file):
    if not imu_data:
        print(f"No IMU data to write for {csv_file}.")
        return
    with open(csv_file, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["timestamp", "gyro_x", "gyro_y", "gyro_z", "acc_x", "acc_y", "acc_z"])
        for rec in imu_data:
            writer.writerow(rec)
    print(f"IMU CSV file written: {csv_file}")

def process_lvx(lvx_file, frames_per_output):
    """Processes the lvx file into 'chunks'.
    All outputs are saved into the same folder as the input lvx file, generates:
    - A .las file for each chunk of 'x' frames (named with the lvx file base name, e.g. sample_0.las)
    - An IMU data .csv for each chunk of 'x' frames
    - An IMU data .csv file of the complete .lvx file"""
    out_dir = os.path.dirname(lvx_file)
    base_name = os.path.splitext(os.path.basename(lvx_file))[0]

    with open(lvx_file, 'rb') as f:
        data = f.read()

    offset = 0
    # Skip headers
    offset += 24  # public header
    frame_duration = struct.unpack('<I', data[offset:offset+4])[0]; offset += 4
    device_count = data[offset]; offset += 1
    offset = 24 + 4 + 1 + device_count * 59

    total_frames = 0
    chunk_index = 0
    overall_imu = []
    chunk_points = []
    chunk_imu = []

    all_points = []
    all_imu = []

    # Process frames
    while offset < len(data):
        if offset + 24 > len(data):
            break
        current_offset = struct.unpack('<Q', data[offset:offset+8])[0]; offset += 8
        next_offset    = struct.unpack('<Q', data[offset:offset+8])[0]; offset += 8
        frame_index    = struct.unpack('<Q', data[offset:offset+8])[0]; offset += 8

        frame_points, frame_imu, offset = process_frame(data, offset, next_offset)

        total_frames += 1

        if frames_per_output == 0:
            all_points.extend(frame_points)
            all_imu.extend(frame_imu)
        else:
            chunk_points.extend(frame_points)
            chunk_imu.extend(frame_imu)
            overall_imu.extend(frame_imu)

            if total_frames % frames_per_output == 0 or offset >= len(data):
                out_las = os.path.join(out_dir, f"0{chunk_index}_{base_name}.las")
                create_las_from_points(chunk_points, out_las)
                out_csv = os.path.join(out_dir, f"{base_name}_imu_chunk_{chunk_index}.csv")
                write_imu_csv(chunk_imu, out_csv)
                print(f"Processed frames {total_frames - frames_per_output} to {total_frames - 1} into chunk {chunk_index}.")
                chunk_index += 1
                chunk_points = []
                chunk_imu = []

    if frames_per_output == 0:
        out_las = os.path.join(out_dir, f"{base_name}_All.las")
        create_las_from_points(all_points, out_las)
        out_csv = os.path.join(out_dir, f"{base_name}_All.csv")
        write_imu_csv(all_imu, out_csv)
        overall_imu = all_imu
        print(f"Processed all {total_frames} frames into one chunk.")

    #Write full length of lvx IMU data to a CSV
    overall_csv = os.path.join(out_dir, f"{base_name}_imu_overall.csv")
    write_imu_csv(overall_imu, overall_csv)
    print(f"Overall IMU data written to {overall_csv}.")

def main():
    parser = argparse.ArgumentParser(
        description="Convert all LVX files in a folder to LAS/IMU chunks."
    )
    parser.add_argument("input_folder", help="Path to the folder containing .lvx files")
    parser.add_argument("frames_per_output", type=int, help="Number of frames per output file")
    args = parser.parse_args()

    if not os.path.isdir(args.input_folder):
        print(f"Folder {args.input_folder} not found or is not a directory.")
        return

    # Iterate through all .lvx files in the folder (for processing multiple scans in one go)
    for fname in sorted(os.listdir(args.input_folder)):
        if fname.lower().endswith(".lvx"):
            lvx_path = os.path.join(args.input_folder, fname)
            print(f"-Processing {lvx_path}")
            process_lvx(lvx_path, args.frames_per_output)

if __name__ == "__main__":
    main()

# To run:
# In your terminal make sure the current directory is where this file is located.
# Type: python lvx_to_las.py [path to folder with .lvx files] [# of frames per chunk]
# Use 0 to process all frames into one las file.