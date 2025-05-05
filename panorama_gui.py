import tkinter as tk
from tkinter import filedialog, ttk, messagebox
import os
import threading
import sys
import time
import cv2
from pathlib import Path
from PIL import Image, ImageTk

# Import main program functionality
from main import capture_key_frames, stitch_images_all_at_once, crop_content

class PanoramaApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Panoramic Image Generator")
        self.root.geometry("800x500")
        self.root.resizable(True, True)
        
        # Set variables
        self.video_path = tk.StringVar()
        self.output_path = tk.StringVar(value="panorama.jpg")
        self.temp_dir = tk.StringVar(value="key_frames")
        self.is_processing = False
        
        # Image related variables
        self.crop_mode = False
        self.crop_start_x = None
        self.crop_start_y = None
        self.crop_end_x = None
        self.crop_end_y = None
        self.crop_rect_id = None
        self.current_image = None
        self.current_photo = None
        self.original_image = None
        
        # Create UI
        self.create_widgets()
        
    def create_widgets(self):
        # Create main layout - control panel on left, image preview on right
        main_pane = ttk.PanedWindow(self.root, orient=tk.HORIZONTAL)
        main_pane.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Left control panel
        control_frame = ttk.Frame(main_pane, padding="10")
        main_pane.add(control_frame, weight=1)
        
        # Title
        title_label = ttk.Label(control_frame, text="Panoramic Generator", font=("Arial", 14, "bold"))
        title_label.pack(pady=(0, 15))
        
        # File selection area
        input_frame = ttk.LabelFrame(control_frame, text="Input/Output", padding="10")
        input_frame.pack(fill=tk.X, pady=5)
        
        # Video file selection
        ttk.Label(input_frame, text="Video:").grid(row=0, column=0, sticky=tk.W, pady=5)
        self.video_entry = ttk.Entry(input_frame, textvariable=self.video_path, width=25)
        self.video_entry.grid(row=0, column=1, padx=5, sticky=tk.EW)
        
        browse_btn = ttk.Button(input_frame, text="Browse", command=self.browse_video, width=8)
        browse_btn.grid(row=0, column=2, padx=5)
        
        # Output file selection
        ttk.Label(input_frame, text="Output:").grid(row=1, column=0, sticky=tk.W, pady=5)
        self.output_entry = ttk.Entry(input_frame, textvariable=self.output_path, width=25)
        self.output_entry.grid(row=1, column=1, padx=5, sticky=tk.EW)
        
        save_btn = ttk.Button(input_frame, text="Browse", command=self.browse_output, width=8)
        save_btn.grid(row=1, column=2, padx=5)
        
        # Configure grid column weights
        input_frame.columnconfigure(1, weight=1)
        
        # Action buttons area
        action_frame = ttk.Frame(control_frame)
        action_frame.pack(fill=tk.X, pady=15)
        
        self.generate_btn = ttk.Button(action_frame, text="Generate Panorama", command=self.generate_panorama)
        self.generate_btn.pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)
        
        self.cancel_btn = ttk.Button(action_frame, text="Cancel", command=self.cancel_process, state=tk.DISABLED)
        self.cancel_btn.pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)
        
        # Progress bar
        self.progress = ttk.Progressbar(control_frame, orient=tk.HORIZONTAL, mode='indeterminate')
        self.progress.pack(fill=tk.X, pady=10)
        
        # Status label
        status_frame = ttk.LabelFrame(control_frame, text="Status")
        status_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        
        # Status text area
        self.status_text = tk.Text(status_frame, height=8, width=30, wrap=tk.WORD)
        self.status_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        scrollbar = ttk.Scrollbar(status_frame, orient=tk.VERTICAL, command=self.status_text.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.status_text.config(yscrollcommand=scrollbar.set)
        
        # Image preview area
        preview_frame = ttk.LabelFrame(main_pane, text="Image Preview")
        main_pane.add(preview_frame, weight=3)  # Set weight to 3 to use more space
        
        # Create Canvas for displaying images and selecting crop area
        self.canvas = tk.Canvas(preview_frame, bg="lightgray", highlightthickness=0)
        self.canvas.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Bind mouse events to Canvas
        self.canvas.bind("<ButtonPress-1>", self.on_crop_start)
        self.canvas.bind("<B1-Motion>", self.on_crop_motion)
        self.canvas.bind("<ButtonRelease-1>", self.on_crop_end)
        self.canvas.bind("<Configure>", self.on_canvas_resize)
        
        # Image operation buttons
        img_btn_frame = ttk.Frame(preview_frame)
        img_btn_frame.pack(fill=tk.X, padx=5, pady=5)
        
        self.crop_btn = ttk.Button(img_btn_frame, text="Crop Mode", command=self.toggle_crop_mode, state=tk.DISABLED, width=10)
        self.crop_btn.pack(side=tk.LEFT, padx=2)
        
        self.apply_crop_btn = ttk.Button(img_btn_frame, text="Apply Crop", command=self.apply_crop, state=tk.DISABLED, width=10)
        self.apply_crop_btn.pack(side=tk.LEFT, padx=2)
        
        self.reset_btn = ttk.Button(img_btn_frame, text="Reset", command=self.reset_image, state=tk.DISABLED, width=8)
        self.reset_btn.pack(side=tk.LEFT, padx=2)
        
        self.save_btn = ttk.Button(img_btn_frame, text="Save As", command=self.save_image_as, state=tk.DISABLED, width=8)
        self.save_btn.pack(side=tk.LEFT, padx=2)
        
        # Initialize status message
        self.add_status("Ready. Please select a video file to begin.")
    
    def on_canvas_resize(self, event):
        """Adjust image size when canvas size changes"""
        if self.current_image is not None:
            self.root.after(100, self.display_current_image)
    
    def browse_video(self):
        filetypes = [("Video Files", "*.mp4 *.avi"), ("All Files", "*.*")]
        file_path = filedialog.askopenfilename(filetypes=filetypes)
        if file_path:
            self.video_path.set(file_path)
            self.add_status(f"Selected: {os.path.basename(file_path)}")
    
    def browse_output(self):
        filetypes = [("JPEG Image", "*.jpg"), ("PNG Image", "*.png"), ("All Files", "*.*")]
        file_path = filedialog.asksaveasfilename(defaultextension=".jpg", filetypes=filetypes)
        if file_path:
            self.output_path.set(file_path)
            self.add_status(f"Output: {os.path.basename(file_path)}")
    
    def add_status(self, message):
        """Add status information to the status area"""
        self.status_text.insert(tk.END, message + "\n")
        self.status_text.see(tk.END)
        self.root.update_idletasks()
    
    def update_ui_for_processing(self, is_processing):
        self.is_processing = is_processing
        state = tk.DISABLED if is_processing else tk.NORMAL
        cancel_state = tk.NORMAL if is_processing else tk.DISABLED
        
        self.video_entry.config(state=state)
        self.output_entry.config(state=state)
        self.generate_btn.config(state=state)
        self.cancel_btn.config(state=cancel_state)
        
        if is_processing:
            self.progress.start(10)
        else:
            self.progress.stop()
    
    def cancel_process(self):
        if self.is_processing:
            self.add_status("Cancelling process...")
    
    def generate_panorama(self):
        video_path = self.video_path.get()
        output_path = self.output_path.get()
        temp_dir = self.temp_dir.get()
        
        # Check video file
        if not video_path or not os.path.exists(video_path):
            messagebox.showerror("Error", "Please select a valid video file")
            return
        
        # Check video format
        ext = os.path.splitext(video_path)[1].lower()
        if ext not in ['.mp4', '.avi']:
            messagebox.showerror("Error", "Please select a valid video format")
            return
        
        # Check output path
        if not output_path:
            messagebox.showerror("Error", "Please set output file path")
            return
        
        # Ensure output directory exists
        output_dir = os.path.dirname(output_path)
        if output_dir and not os.path.exists(output_dir):
            try:
                os.makedirs(output_dir)
            except Exception as e:
                messagebox.showerror("Error", f"Cannot create output directory: {str(e)}")
                return
        
        # Update UI status
        self.update_ui_for_processing(True)
        
        # Run processing in a separate thread
        threading.Thread(
            target=self.process_panorama,
            args=(video_path, output_path, temp_dir),
            daemon=True
        ).start()
    
    def process_panorama(self, video_path, output_path, temp_dir):
        try:
            self.add_status(f"Processing video...")
            start_time = time.time()
            
            # Step 1: Capture key frames
            self.add_status("Step 1/3: Capturing key frames...")
            frame_count = capture_key_frames(video_path, temp_dir)
            
            if frame_count <= 1:
                self.add_status("Not enough key frames captured")
                self.root.after(0, lambda: self.process_complete(False, "Not enough key frames"))
                return
            
            # Step 2: Stitch key frames
            self.add_status(f"Step 2/3: Stitching {frame_count} frames...")
            frames = sorted([os.path.join(temp_dir, f) for f in os.listdir(temp_dir) 
                           if f.startswith('frame') and f.endswith('.jpg')])
            
            if not frames:
                self.add_status(f"No key frames found")
                self.root.after(0, lambda: self.process_complete(False, "No key frames found"))
                return
            
            pano = stitch_images_all_at_once(frames)
            
            if pano is None:
                self.add_status("Stitching failed")
                self.root.after(0, lambda: self.process_complete(False, "Stitching failed"))
                return
            
            # Step 3: Crop black borders and save result
            self.add_status("Step 3/3: Cropping and saving...")
            pano = crop_content(pano)
            
            # Save result
            cv2.imwrite(output_path, pano)
            self.add_status(f"Panorama saved successfully")
            
            # Clean up temporary files
            self.add_status(f"Cleaning up...")
            import shutil
            shutil.rmtree(temp_dir)
            
            elapsed_time = time.time() - start_time
            self.add_status(f"Complete! Took {elapsed_time:.1f} seconds")
            
            # Update UI in main thread
            self.root.after(0, lambda: self.process_complete(True))
            
        except Exception as e:
            error_msg = str(e)
            self.add_status(f"Error: {error_msg}")
            self.root.after(0, lambda: self.process_complete(False, error_msg))
    
    def process_complete(self, success, error_msg=None):
        self.update_ui_for_processing(False)
        
        if success:
            messagebox.showinfo("Success", "Panoramic image generated successfully")
            # Display generated image
            output_path = self.output_path.get()
            if os.path.exists(output_path):
                self.display_image(output_path)
        else:
            messagebox.showerror("Error", f"Processing failed: {error_msg}")
    
    def display_image(self, image_path):
        try:
            # Open image with PIL
            img = Image.open(image_path)
            
            # Save original image
            self.original_image = img.copy()
            self.current_image = img.copy()
            
            # Display image
            self.display_current_image()
            
            # Enable crop and save buttons
            self.crop_btn.config(state=tk.NORMAL)
            self.save_btn.config(state=tk.NORMAL)
            
            # Add status message
            self.add_status(f"Image loaded: {os.path.basename(image_path)}")
            
        except Exception as e:
            self.add_status(f"Cannot display image: {str(e)}")
    
    def display_current_image(self):
        """Display current image on canvas"""
        if self.current_image is None:
            return
        
        # Get canvas size
        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()
        
        # Use default values if canvas is not yet rendered
        if canvas_width <= 1:
            canvas_width = 600
        if canvas_height <= 1:
            canvas_height = 400
        
        # Calculate scale ratio to fit canvas
        img_width, img_height = self.current_image.size
        scale = min(canvas_width / img_width, canvas_height / img_height)
        
        # Calculate new dimensions
        new_width = int(img_width * scale)
        new_height = int(img_height * scale)
        
        # Resize image
        resized_img = self.current_image.resize((new_width, new_height), Image.LANCZOS)
        
        # Convert to Tkinter-compatible image object
        self.current_photo = ImageTk.PhotoImage(resized_img)
        
        # Clear canvas
        self.canvas.delete("all")
        
        # Display image at center of canvas
        x = (canvas_width - new_width) // 2
        y = (canvas_height - new_height) // 2
        self.canvas.create_image(x, y, anchor=tk.NW, image=self.current_photo)
    
    def toggle_crop_mode(self):
        """Toggle crop mode"""
        self.crop_mode = not self.crop_mode
        if self.crop_mode:
            self.crop_btn.config(text="Cancel Crop")
            self.add_status("Drag to select crop area")
        else:
            self.crop_btn.config(text="Crop Mode")
            # Clear crop rectangle
            if self.crop_rect_id:
                self.canvas.delete(self.crop_rect_id)
                self.crop_rect_id = None
            self.crop_start_x = None
            self.crop_start_y = None
            self.crop_end_x = None
            self.crop_end_y = None
            self.apply_crop_btn.config(state=tk.DISABLED)
    
    def on_crop_start(self, event):
        """Start crop selection"""
        if not self.crop_mode or self.current_image is None:
            return
        # Record start position
        self.crop_start_x = self.canvas.canvasx(event.x)
        self.crop_start_y = self.canvas.canvasy(event.y)
        # Clear previous crop box
        if self.crop_rect_id:
            self.canvas.delete(self.crop_rect_id)
            self.crop_rect_id = None
    
    def on_crop_motion(self, event):
        """Update crop box during drag"""
        if not self.crop_mode or self.crop_start_x is None or self.current_image is None:
            return
        # Update end position
        self.crop_end_x = self.canvas.canvasx(event.x)
        self.crop_end_y = self.canvas.canvasy(event.y)
        # Clear previous crop box
        if self.crop_rect_id:
            self.canvas.delete(self.crop_rect_id)
        # Draw new crop box
        self.crop_rect_id = self.canvas.create_rectangle(
            self.crop_start_x, self.crop_start_y, 
            self.crop_end_x, self.crop_end_y,
            outline="red", width=2
        )
    
    def on_crop_end(self, event):
        """Complete crop selection"""
        if not self.crop_mode or self.crop_start_x is None or self.current_image is None:
            return
        # Update end position
        self.crop_end_x = self.canvas.canvasx(event.x)
        self.crop_end_y = self.canvas.canvasy(event.y)
        # Ensure there is a valid selection area
        if (abs(self.crop_end_x - self.crop_start_x) > 10 and 
            abs(self.crop_end_y - self.crop_start_y) > 10):
            self.apply_crop_btn.config(state=tk.NORMAL)
            self.add_status("Click 'Apply Crop' to crop")
        else:
            if self.crop_rect_id:
                self.canvas.delete(self.crop_rect_id)
                self.crop_rect_id = None
            self.crop_start_x = None
            self.crop_start_y = None
            self.crop_end_x = None
            self.crop_end_y = None
            self.apply_crop_btn.config(state=tk.DISABLED)
    
    def apply_crop(self):
        """Apply crop"""
        if self.current_image is None or None in (self.crop_start_x, self.crop_start_y, self.crop_end_x, self.crop_end_y):
            return
        
        # Get size ratio between image and canvas to convert crop coordinates
        img_width, img_height = self.current_image.size
        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()
        
        # Calculate actual position and size of image in Canvas
        scale = min(canvas_width / img_width, canvas_height / img_height)
        new_width = int(img_width * scale)
        new_height = int(img_height * scale)
        dx = (canvas_width - new_width) // 2
        dy = (canvas_height - new_height) // 2
        
        # Convert Canvas coordinates to image coordinates
        x1 = max(0, int((self.crop_start_x - dx) / scale))
        y1 = max(0, int((self.crop_start_y - dy) / scale))
        x2 = min(img_width, int((self.crop_end_x - dx) / scale))
        y2 = min(img_height, int((self.crop_end_y - dy) / scale))
        
        # Ensure x1 < x2, y1 < y2
        if x1 > x2:
            x1, x2 = x2, x1
        if y1 > y2:
            y1, y2 = y2, y1
        
        # Crop image
        cropped_image = self.current_image.crop((x1, y1, x2, y2))
        
        # Update image
        self.current_image = cropped_image
        self.display_current_image()
        
        # Update UI
        self.crop_mode = False
        self.crop_btn.config(text="Crop Mode")
        if self.crop_rect_id:
            self.canvas.delete(self.crop_rect_id)
            self.crop_rect_id = None
        self.crop_start_x = None
        self.crop_start_y = None
        self.crop_end_x = None
        self.crop_end_y = None
        self.apply_crop_btn.config(state=tk.DISABLED)
        self.reset_btn.config(state=tk.NORMAL)
        
        self.add_status(f"Image cropped to {cropped_image.width}x{cropped_image.height}")
    
    def reset_image(self):
        """Reset image to original state"""
        if self.original_image is not None:
            self.current_image = self.original_image.copy()
            self.display_current_image()
            self.add_status("Image reset to original")
    
    def save_image_as(self):
        """Save current image as"""
        if self.current_image is None:
            return
        
        # Open save dialog
        filetypes = [
            ("JPEG Image", "*.jpg"),
            ("PNG Image", "*.png"),
            ("All Files", "*.*")
        ]
        file_path = filedialog.asksaveasfilename(defaultextension=".jpg", filetypes=filetypes)
        
        if file_path:
            try:
                self.current_image.save(file_path)
                self.add_status(f"Saved to: {os.path.basename(file_path)}")
            except Exception as e:
                self.add_status(f"Save error: {str(e)}")
                messagebox.showerror("Save Error", f"Cannot save image: {str(e)}")

if __name__ == "__main__":
    root = tk.Tk()
    app = PanoramaApp(root)
    root.mainloop() 