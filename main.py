import cv2
import numpy as np
import os
import argparse
import shutil
import time
import copy

def capture_key_frames(video_path, output_dir='key_frames'):
    # Ensure output directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Create VideoCapture object to get video stream
    vid_cap = cv2.VideoCapture(video_path)
    
    # Get total frame count and frame rate
    total_frames = int(vid_cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = vid_cap.get(cv2.CAP_PROP_FPS)
    print(f"Total frames: {total_frames}, Frame rate: {fps}")

    # Use SIFT descriptors to describe overlap areas between current and adjacent frames
    sift = cv2.SIFT_create()

    # Select the first frame as key frame by default
    success, last = vid_cap.read()
    cv2.imwrite(f'{output_dir}/frame0.jpg', last)
    print("Captured frame0.jpg")
    count = 1
    frame_num = 1

    w = int(last.shape[1] * 2 / 3)  # Region for detecting matching points
    step = 40          # Step size for accelerating capture
    min_match_num = 100  # Minimum number of matches required (for good stitching)
    max_match_num = 1000  # Maximum number of matches (to avoid redundant frames)
    
    # Force capture variables
    force_capture_interval = 100  # Frames
    last_capture_frame = 0

    # Read next frame
    success, image = vid_cap.read()
    
    while success:
        # Display processing progress
        if count % 50 == 0:
            print(f"Processing progress: {count}/{total_frames} ({count/total_frames*100:.1f}%)")
            
        force_capture = (count - last_capture_frame >= force_capture_interval)
        
        if count % step == 0:
            try:
                # Detect and compute keypoints and descriptors
                kp1, des1 = sift.detectAndCompute(last[:, -w:], None)
                kp2, des2 = sift.detectAndCompute(image[:, :w], None)
                
                capture_this_frame = False
                inliers = 0
                
                if des1 is not None and des2 is not None and len(des1) > 0 and len(des2) > 0:
                    # Use brute force matcher to get matches
                    bf = cv2.BFMatcher(normType=cv2.NORM_L2)
                    matches = bf.knnMatch(des1, des2, k=2)
                    
                    if len(matches) > 0:
                        # Define valid match: distance less than match_ratio times the distance of the second best match
                        match_ratio = 0.8
                        
                        # Select valid matches
                        valid_matches = []
                        for m in matches:
                            if len(m) == 2:
                                m1, m2 = m
                                if m1.distance < match_ratio * m2.distance:
                                    valid_matches.append(m1)
                        
                        # At least 4 points needed to calculate homography matrix
                        if len(valid_matches) > 4:
                            img1_pts = []
                            img2_pts = []
                            for match in valid_matches:
                                img1_pts.append(kp1[match.queryIdx].pt)
                                img2_pts.append(kp2[match.trainIdx].pt)
                            
                            # Format as matrix (for homography calculation)
                            img1_pts = np.float32(img1_pts).reshape(-1, 1, 2)
                            img2_pts = np.float32(img2_pts).reshape(-1, 1, 2)
                            
                            # Calculate homography matrix
                            _, mask = cv2.findHomography(img1_pts, img2_pts,
                                                        cv2.RANSAC, 5.0)
                            
                            if mask is not None:
                                inliers = np.count_nonzero(mask)
                                
                                if min_match_num < inliers < max_match_num:
                                    capture_this_frame = True
                
                # If feature-based method cannot capture this frame but force capture interval is exceeded, force capture
                if force_capture:
                    capture_this_frame = True
                
                if capture_this_frame:
                    # Save key frame as JPG file
                    last = image.copy()
                    print(f"Captured frame{frame_num}.jpg")
                    cv2.imwrite(f'{output_dir}/frame{frame_num}.jpg', last)
                    frame_num += 1
                    last_capture_frame = count
                    
            except Exception as e:
                print(f"Error processing frame {count}: {e}")
        
        success, image = vid_cap.read()
        count += 1
    
    print(f"Processing complete. Captured {frame_num} key frames.")
    vid_cap.release()
    return frame_num



def stitch_images_all_at_once(image_paths):

    # Read all images
    images = []
    for path in image_paths:
        img = cv2.imread(path)
        if img is None:
            print(f"Cannot read image: {path}")
            continue
        images.append(img)
    
    if len(images) < 2:
        print("At least two images are required for stitching")
        return None
    
    # Create stitcher
    stitcher = cv2.Stitcher.create(cv2.Stitcher_PANORAMA)
    
    # Perform stitching
    print(f"Starting to stitch {len(images)} images at once...")
    status, pano = stitcher.stitch(images)
    
    if status != cv2.Stitcher_OK:
        error_messages = {
            cv2.Stitcher_ERR_NEED_MORE_IMGS: "Need more images",
            cv2.Stitcher_ERR_HOMOGRAPHY_EST_FAIL: "Homography estimation failed",
            cv2.Stitcher_ERR_CAMERA_PARAMS_ADJUST_FAIL: "Camera parameter adjustment failed"
        }
        print(f"Stitching failed: {error_messages.get(status, f'Unknown error {status}')}")
        return None
    
    return pano

def crop_content(image):
    """Crop black borders from image, keeping only the valid content area"""
    if image is None:
        return None
    
    # Convert to grayscale and binarize
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
    
    # Get image dimensions
    height, width = binary.shape
    
    # Scan from all four boundaries inward to find the edges of non-black regions
    # Scan top boundary
    top = 0
    for i in range(height):
        if np.sum(binary[i, :]) > 0:
            top = i
            break
    
    # Scan bottom boundary
    bottom = height - 1
    for i in range(height - 1, -1, -1):
        if np.sum(binary[i, :]) > 0:
            bottom = i
            break
    
    # Scan left boundary
    left = 0
    for i in range(width):
        if np.sum(binary[:, i]) > 0:
            left = i
            break
    
    # Scan right boundary
    right = width - 1
    for i in range(width - 1, -1, -1):
        if np.sum(binary[:, i]) > 0:
            right = i
            break
    
    # Safety check - ensure crop region is valid
    if left >= right or top >= bottom:
        print("Invalid crop region, returning original image")
        return image
    
    # Crop image
    cropped_image = image[top:bottom+1, left:right+1]
    
    print(f"Original image size: {width}x{height}")
    print(f"Cropped image size: {cropped_image.shape[1]}x{cropped_image.shape[0]}")
    print(f"Crop region: left={left}, right={right}, top={top}, bottom={bottom}")
    
    return cropped_image

def show_orb(img1_path, img2_path, output_path='orb_matches.jpg'):

    # Read images
    img1 = cv2.imread(img1_path)
    img2 = cv2.imread(img2_path)
    
    if img1 is None or img2 is None:
        print(f"Cannot read images: {img1_path} or {img2_path}")
        return 0
    
    # Create ORB detector
    orb = cv2.ORB_create(nfeatures=1000)
    
    # Detect keypoints and compute descriptors
    kp1, des1 = orb.detectAndCompute(img1, None)
    kp2, des2 = orb.detectAndCompute(img2, None)
    
    print(f"Detected {len(kp1)} feature points in image 1")
    print(f"Detected {len(kp2)} feature points in image 2")
    
    # Create BF matcher
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    
    # Perform matching
    matches = bf.match(des1, des2)
    
    # Sort by distance
    matches = sorted(matches, key=lambda x: x.distance)
    
    # Only draw best matches (top 50)
    good_matches = matches[:min(50, len(matches))]
    
    # Draw matching results
    matched_img = cv2.drawMatches(img1, kp1, img2, kp2, good_matches, None, 
                                  flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    
    # Save result
    cv2.imwrite(output_path, matched_img)
    print(f"Matching result saved as {output_path}")
    print(f"Found {len(matches)} matching points, displaying the best {len(good_matches)} matches")
    
    return len(matches)

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Generate panoramic image from video file')
    parser.add_argument('video', help='Input video file path')
    parser.add_argument('--output', default='panorama.jpg', help='Output panorama filename')
    #parser.add_argument('--temp_dir', default='key_frames', help='Directory for storing temporary key frames')
    #parser.add_argument('--keep_frames', action='store_true', help='Keep captured key frames instead of deleting them')
    #parser.add_argument('--test_orb', action='store_true', help='Test ORB feature matching (requires two key frames)')
    args = parser.parse_args()
    
    start_time = time.time()
    
    # Step 1: Capture key frames
    print(f"Capturing key frames from video {args.video}...")
    frame_count = capture_key_frames(args.video, args.temp_dir)
    
    if frame_count <= 1:
        print("Not enough key frames captured for stitching")
        return
    
    # Test ORB feature matching
    #if args.test_orb:
    #    frames = sorted([os.path.join(args.temp_dir, f) for f in os.listdir(args.temp_dir) 
    #                if f.startswith('frame') and f.endswith('.jpg')])
    #    if len(frames) >= 2:
    #        print("\nTesting ORB feature matching...")
    #        show_orb(frames[0], frames[1], 'orb_matches.jpg')
    #    else:
    #        print("Not enough key frames for ORB testing")
    
    #  Stitch key frames
    print("\nStarting to stitch key frames...")
    frames = sorted([os.path.join(args.temp_dir, f) for f in os.listdir(args.temp_dir) 
                    if f.startswith('frame') and f.endswith('.jpg')])
    
    if not frames:
        print(f"No key frames found in {args.temp_dir}")
        return
    
    print(f"Found {len(frames)} key frames, starting stitching")
    pano = stitch_images_all_at_once(frames)
    
    # Save result
    if pano is not None:
        # Crop black edges
        print("Cropping black edges...")
        pano = crop_content(pano)
        
        # Save result
        cv2.imwrite(args.output, pano)
        print(f"Panorama image saved as {args.output}")
    else:
        print("Stitching failed")
    
    # Clean up temporary files
    if not args.keep_frames:
        print(f"Cleaning up temporary directory {args.temp_dir}...")
        shutil.rmtree(args.temp_dir)
    else:
        print(f"Keeping key frames in {args.temp_dir} directory")
    
    elapsed_time = time.time() - start_time
    print(f"Processing complete, took {elapsed_time:.1f} seconds")


if __name__ == "__main__":
    main() 