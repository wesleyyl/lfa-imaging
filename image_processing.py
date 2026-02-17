import numpy as np
import cv2
import matplotlib.pyplot as plt
from pathlib import Path


class SimpleLFAAnalyzer:
    """
    Simple LFA Analyzer for cropped images (detection zone only, no wick)

    Works by:
    1. Splitting image in half (top = control, bottom = test)
    2. Finding darkest line in each half (pink lines are darker than white background)
    3. Measuring intensity and calculating relative intensity
    """

    def __init__(self, image_path):
        """Initialize with image path"""
        self.image_path = image_path
        self.original_image = cv2.imread(str(image_path))
        if self.original_image is None:
            raise ValueError(f"Could not load image from {image_path}")

        self.gray_image = None
        self.inverted_image = None
        self.top_half = None
        self.bottom_half = None
        self.control_line_pos = None
        self.test_line_pos = None
        self.control_intensity = None
        self.test_intensity = None
        self.background = None
        self.relative_intensity = None
        self.is_negative = False

        #WL:
        self.background_image = None
        self.corrected_image = None
        self.otsu_threshold = None
        self.binary_mask = None

    def preprocess(self):
        """Convert to grayscale and invert (so pink lines have HIGH values)"""
        # Grayscale
        self.gray_image = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2GRAY)

        # Invert: pink lines are darker than white background
        # After inversion: pink lines will be BRIGHT
        self.inverted_image = 255 - self.gray_image

        return self.inverted_image

    def split_halves(self, use_corrected=False):
        """Split image into top (control) and bottom (test) halves"""

        if use_corrected:
          if self.corrected_image is None:
            raise ValueError("Run subtract_background() first or set use_corrected=False.")
          img = self.corrected_image
        else:
          img = self.inverted_image

        height = img.shape[0]
        mid_point = height // 2
        # Skip first 3 pixels of each half to avoid any marking lines
        skip = 3

        self.top_half = img[skip:mid_point-skip, :]
        self.bottom_half = img[mid_point+skip:height-skip, :]

        return self.top_half, self.bottom_half

    def find_darkest_line(self, image_half, half_name=""):
        """
        Find the darkest horizontal line (highest intensity after inversion)

        Returns: (line_position, line_intensity, intensity_profile)
        """
        # Average intensity across each row
        intensity_profile = np.mean(image_half, axis=1)

        # Find maximum intensity (darkest line in original)
        max_pos = np.argmax(intensity_profile)
        max_intensity = intensity_profile[max_pos]

        return max_pos, max_intensity, intensity_profile

    def subtract_background(self, method="morph", ksize=51, normalize=False, denoise=False):
        """
        Background subtraction on the inverted image.

        Parameters
        ----------
        method : str
            "morph" = morphological opening (recommended)
            "blur"  = large Gaussian blur background estimate
        ksize : int
            Kernel size for background estimation (odd is best).
            Larger = more aggressive background removal.

        Returns
        -------
        corrected : np.ndarray (uint8)
            Background-subtracted version of inverted image.
        """

        if self.inverted_image is None:
            raise ValueError("Run preprocess() first.")

        img = self.inverted_image.astype(np.uint8)

        if method == "tophat":
            print("RUNNING TEST")
            # BETTER KERNEL FOR HORIZONTAL LINES
            kernel = cv2.getStructuringElement(
                cv2.MORPH_RECT,
                # (ksize * 3, ksize)   # wide horizontally
                (max(3, int(ksize * 3)), max(3, int(ksize)))
            )

            # Use white top-hat instead of manual subtract
            corrected = cv2.morphologyEx(img, cv2.MORPH_TOPHAT, kernel)

            # If you still want background saved:
            background = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
        
        elif method == "morph":
            # Large kernel removes thin bright lines, keeps smooth background
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
            background = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
            corrected = cv2.subtract(img, background)  # clips at 0 automatically for uint8
            
        elif method == "morph_ellips_mod":
            kh = ksize                  # height
            kw = int(ksize * 1.3)         # width multiplier (3–10 is typical)

            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kw, kh))
            background = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
            corrected = cv2.subtract(img, background)

        elif method == "morph_rect":
            # Horizontal-aware opening kernel
            kh = ksize                      # height
            kw = ksize * 5                  # width (tune 3–10×)
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kw, kh))

            background = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
            corrected = cv2.subtract(img, background)
        elif method == "blur":
            # Smooth illumination background estimate
            if ksize % 2 == 0:
                ksize += 1  # GaussianBlur prefers odd
            background = cv2.GaussianBlur(img, (ksize, ksize), 0)
            corrected = cv2.subtract(img, background)  # clips at 0 automatically for uint8

        else:
            raise ValueError("method must be 'morph' or 'blur' or 'tophat'")
      
      
         # Post-processing AFTER corrected is finalized
        if normalize:
            corrected = cv2.normalize(corrected, None, 0, 255, cv2.NORM_MINMAX)

        if denoise:
            corrected = cv2.medianBlur(corrected, 3)

        self.background_image = background
        self.corrected_image = corrected.astype(np.uint8)
        return self.corrected_image

    def otsu_binarize_corrected(self, blur_ksize=2, invert_mask=False, otsu_scale=0.75):
        """
        Apply Otsu thresholding to the background-subtracted (corrected) image
        to separate foreground (line) vs background/noise.

        Parameters
        ----------
        blur_ksize : int
            Optional Gaussian blur kernel size before Otsu (odd). Helps reduce speckle.
            Set to 0 or 1 to skip blurring.
        invert_mask : bool
            If True, invert the binary mask (swap foreground/background).

        Returns
        -------
        thresh : float
            Otsu threshold value.
        mask : np.ndarray (uint8)
            Binary mask, values in {0,255}.
        """
        if self.corrected_image is None:
            raise ValueError("Run preprocess() and subtract_background() first.")

        img = self.corrected_image.copy()

        # Optional denoise to make histogram bimodality cleaner
        if blur_ksize and blur_ksize > 1:
            if blur_ksize % 2 == 0:
                blur_ksize += 1
            img = cv2.GaussianBlur(img, (blur_ksize, blur_ksize), 0)

        # Otsu threshold
        # Foreground in corrected image should be BRIGHT (since inverted), so THRESH_BINARY is appropriate
        # t, mask = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        t_otsu, _ = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Make threshold less aggressive
        t_used = max(0.0, float(otsu_scale) * float(t_otsu))

        # Apply threshold
        _, mask = cv2.threshold(img, t_used, 255, cv2.THRESH_BINARY)

        if invert_mask:
            mask = cv2.bitwise_not(mask)

        self.otsu_threshold = float(t_used)
        self.binary_mask = mask
        return self.otsu_threshold, self.binary_mask
    
    
    def _median_filter_1d(self, x, ksize):
        """
        1D median filter with reflect padding.
        x: (H,) float array
        ksize: odd int >= 3
        """
        k = int(ksize)
        if k < 3:
            k = 3
        if k % 2 == 0:
            k += 1

        pad = k // 2
        xp = np.pad(x, pad_width=pad, mode="reflect")
        # sliding window view -> median over last axis
        w = np.lib.stride_tricks.sliding_window_view(xp, window_shape=k)
        return np.median(w, axis=1)
    
    def _keep_true_runs(self, hits, min_run=3):
        """
        Keep only contiguous True runs of length >= min_run.
        hits: (H,) bool
        """
        hits = hits.astype(bool)
        out = np.zeros_like(hits, dtype=bool)
        h = len(hits)

        i = 0
        while i < h:
            if not hits[i]:
                i += 1
                continue
            j = i
            while j < h and hits[j]:
                j += 1
            if (j - i) >= min_run:
                out[i:j] = True
            i = j
        return out

    def _expand_rows(self, hits, radius=2):
        """
        Expand True rows by +/- radius using 1D dilation.
        """
        hits = hits.astype(np.uint8)
        k = 2 * int(radius) + 1
        kernel = np.ones(k, dtype=np.uint8)
        expanded = np.convolve(hits, kernel, mode="same") > 0
        return expanded
    
    def detect_band_rows(
        self,
        img,
        stat="median",
        smooth_ksize=51,
        exclude_center_frac=0.20,
        k=4.0,
    ):
        """
        Build a per-row adaptive threshold from row statistics.

        Idea:
        row_score[i]   = mean/median intensity of row i
        baseline[i]    = smoothed(row_score)  (captures slow drift / illumination gradient)
        resid[i]       = row_score[i] - baseline[i]
        sigma          = robust scale estimate from resid in "background rows"
        T_row[i]       = baseline[i] + k*sigma

        Returns
        -------
        row_score : (H,) float
        baseline  : (H,) float
        T_row     : (H,) float
        row_hits  : (H,) bool   (rows whose score exceeds their threshold)
        peak_row  : int
        sigma     : float
        """
        
        img = img.astype(np.float32)

        # Row statistic
        if stat == "mean":
            row_score = np.mean(img, axis=1)
        elif stat == "median":
            row_score = np.median(img, axis=1)
        else:
            raise ValueError("stat must be 'mean' or 'median'")

        h = row_score.shape[0]

        # Smooth row_score to get a drift baseline
        ksz = int(smooth_ksize)
        if ksz < 3:
            ksz = 3
        if ksz % 2 == 0:
            ksz += 1
        # baseline = cv2.medianBlur(row_score[:, None].astype(np.float32), ksz).ravel()
        baseline = self._median_filter_1d(row_score, smooth_ksize)
        
        # Robust sigma from residuals on "background rows"
        resid = row_score - baseline

        center = h // 2
        band = int(exclude_center_frac * h)
        mask_rows = np.ones(h, dtype=bool)
        mask_rows[max(0, center - band // 2):min(h, center + band // 2)] = False

        bg_resid = resid[mask_rows]
        med = np.median(bg_resid)
        mad = np.median(np.abs(bg_resid - med)) + 1e-6
        sigma = 1.4826 * mad

        # Per-row threshold
        T_row = baseline + float(k) * float(sigma)

        row_hits = row_score > T_row
        peak_row = int(np.argmax(row_score))

        return row_score, baseline, T_row, row_hits, peak_row, float(sigma)
    
    
    def rowwise_binarize_corrected(
        self,
        stat="median",
        smooth_ksize=51,
        exclude_center_frac=0.20,
        k=4.0,
        min_run=3,              # minimum consecutive rows to keep as a band
        expand=2,               # expand hits by +/- expand rows (thicken band)
        invert_mask=False
    ):
        """
        ROW-WISE thresholding: each row is either ON (255 across entire width) or OFF.

        mask[i, :] = 255 if row_score[i] > T_row[i], else 0

        Extra cleanup:
        - keep only runs of True rows with length >= min_run
        - expand bands vertically by +/- expand rows
        """
        if self.corrected_image is None:
            raise ValueError("Run preprocess() and subtract_background() first.")

        img = self.corrected_image.astype(np.float32)

        row_score, baseline, T_row, row_hits, peak_row, sigma = self.detect_band_rows(
            img,
            stat=stat,
            smooth_ksize=smooth_ksize,
            exclude_center_frac=exclude_center_frac,
            k=k,
        )

        # --- keep only long-enough contiguous runs (remove isolated hit rows) ---
        hits = row_hits.copy()
        if min_run and min_run > 1:
            hits = self._keep_true_runs(hits, min_run=min_run)

        # --- expand vertically (optional) ---
        if expand and expand > 0:
            hits = self._expand_rows(hits, radius=expand)

        # --- build ROW-WISE mask (entire row on/off) ---
        mask = (hits[:, None].astype(np.uint8) * 255)
        mask = np.repeat(mask, self.corrected_image.shape[1], axis=1)

        if invert_mask:
            mask = cv2.bitwise_not(mask)

        self.binary_mask = mask
        self.otsu_threshold = None
        self._rowwise_debug = {
            "row_score": row_score,
            "baseline": baseline,
            "T_row": T_row,
            "row_hits": row_hits,
            "row_hits_clean": hits,
            "peak_row": peak_row,
            "sigma": sigma,
            "stat": stat,
            "smooth_ksize": smooth_ksize,
            "k": k,
            "min_run": min_run,
            "expand": expand,
        }
        return T_row, mask
        
    
    def plot_histogram_and_otsu(self, bins=60, show_overlay=True):
        """
        Plot histograms of inverted and corrected pixel intensities and show Otsu threshold.
        Uses thick connected bars (not thin spiky lines).
        Also displays the corrected image and the resulting binary mask.
        """
        if self.inverted_image is None:
            raise ValueError("Run preprocess() first.")
        if self.corrected_image is None:
            raise ValueError("Run subtract_background() first.")
        if self.binary_mask is None or self.otsu_threshold is None:
            # compute Otsu if not already computed
            self.otsu_binarize_corrected()

        inv = self.inverted_image.ravel()
        cor = self.corrected_image.ravel()

        fig, axes = plt.subplots(1, 4, figsize=(20, 4))

        # ----------------------------
        # Histogram: inverted (raw)
        # ----------------------------
        axes[0].hist(
            inv,
            bins=bins,
            density=True,
            histtype='bar',
            edgecolor=None,
            linewidth=0,
            alpha=0.85
        )
        
        # Show same numeric threshold for reference
        axes[0].axvline(
            self.otsu_threshold,
            linestyle="--",
            linewidth=2,
            color="red",
            label=f"Otsu T (corrected) = {self.otsu_threshold:.1f}"
        )
        
        axes[0].set_title("Histogram: Inverted (raw)")
        axes[0].set_xlabel("Pixel intensity")
        axes[0].set_ylabel("Density")

        # ----------------------------
        # Histogram: corrected + Otsu
        # ----------------------------
        axes[1].hist(
            cor,
            bins=bins,
            density=True,
            histtype='bar',
            edgecolor=None,
            linewidth=0,
            alpha=0.85
        )

        axes[1].axvline(
            self.otsu_threshold,
            linestyle="--",
            linewidth=2,
            label=f"Otsu T = {self.otsu_threshold:.1f}"
        )

        axes[1].set_title("Histogram: Corrected (bg-subtracted)")
        axes[1].set_xlabel("Pixel intensity")
        axes[1].legend()

        # ----------------------------
        # Show corrected + mask overlay
        # ----------------------------
        # Panel 3: Overlay
        axes[2].imshow(self.corrected_image, cmap="hot")
        axes[2].imshow(self.binary_mask, alpha=0.35)
        axes[2].set_title("Corrected + Otsu Overlay")
        axes[2].axis("off")

        if show_overlay:
            axes[2].imshow(self.binary_mask, alpha=0.35)


                # Panel 4: Clean binary visualization
        binary_vis = cv2.bitwise_not(self.binary_mask)

        # Add black border for visibility
        bordered = cv2.copyMakeBorder(
            binary_vis,
            top=2, bottom=2, left=2, right=2,
            borderType=cv2.BORDER_CONSTANT,
            value=0  # black border
        )

        axes[3].imshow(bordered, cmap="gray", vmin=0, vmax=255)
        axes[3].set_title("Binarized (Lines Dark)")
        axes[3].axis("off")


        plt.tight_layout()
        plt.show()
        # plt.savefig("debug_plot.png", dpi=300, bbox_inches="tight")
        # print("Saved debug_plot.png")

        return fig
    
    def plot_rowwise_threshold_debug(self, bins=60, show_overlay=True):
        if self.corrected_image is None:
            raise ValueError("Run subtract_background() first.")
        if not hasattr(self, "_rowwise_debug"):
            raise ValueError("Run rowwise_binarize_corrected() first.")

        dbg = self._rowwise_debug
        row_score = dbg["row_score"]
        baseline = dbg["baseline"]
        T_row = dbg["T_row"]
        sigma = dbg["sigma"]

        fig, axes = plt.subplots(1, 4, figsize=(20, 4))

        # 1) Histogram of row scores (not pixels)
        axes[0].hist(row_score, bins=bins, density=True, histtype="bar", alpha=0.85)
        # show a "typical" threshold value band (baseline varies, so show median baseline + k*sigma)
        typical_T = float(np.median(baseline) + dbg["k"] * sigma)
        axes[0].axvline(typical_T, linestyle="--", linewidth=2, color="red",
                        label=f"Typical T ≈ {typical_T:.1f}")
        axes[0].set_title(f"Histogram: Row {dbg['stat']} scores")
        axes[0].set_xlabel("Row score")
        axes[0].set_ylabel("Density")
        axes[0].legend()

        # 2) Row score trace + baseline + threshold
        y = np.arange(len(row_score))
        axes[1].plot(row_score, y, linewidth=1.5, label="row_score")
        axes[1].plot(baseline, y, linewidth=1.5, label="baseline (smoothed)")
        axes[1].plot(T_row, y, linewidth=1.5, label="T_row = baseline + k*sigma")
        axes[1].invert_yaxis()
        axes[1].set_title("Row-wise threshold profile")
        axes[1].set_xlabel("Value")
        axes[1].set_ylabel("Row index")
        axes[1].legend()

        # 3) Corrected + mask overlay
        axes[2].imshow(self.corrected_image, cmap="hot")
        if show_overlay and self.binary_mask is not None:
            axes[2].imshow(self.binary_mask, alpha=0.35)
        axes[2].set_title("Corrected + Row-wise Mask Overlay")
        axes[2].axis("off")

        # 4) Binary visualization (lines dark)
        if self.binary_mask is None:
            raise ValueError("No binary_mask found. Run rowwise_binarize_corrected().")
        binary_vis = cv2.bitwise_not(self.binary_mask)
        bordered = cv2.copyMakeBorder(binary_vis, 2, 2, 2, 2, cv2.BORDER_CONSTANT, value=0)
        axes[3].imshow(bordered, cmap="gray", vmin=0, vmax=255)
        axes[3].set_title("Row-wise Binarized (Lines Dark)")
        axes[3].axis("off")

        plt.tight_layout()
        plt.show()
        return fig
    
    
    
    
    def analyze(self, min_test_threshold=3.0, bg='', k=51, normalize=False, denoise=False, binarize_mode="rowwise"):
        """
        Run complete analysis

        Parameters:
        -----------
        min_test_threshold : float
            Minimum signal above background to consider test line present

        Returns:
        --------
        dict : Results
        """
        print("=" * 70)
        print(f"Analyzing: {Path(self.image_path).name}")
        print("=" * 70)

        # Step 1: Preprocess
        self.preprocess()
        height, width = self.inverted_image.shape
        print(f"Image size: {height} x {width} pixels")
        # Now the image is converted to grayscale and inverted (dark lines become bright)

        # WL: Run background subtraction method
        self.subtract_background(method=bg, ksize=k, normalize=normalize, denoise=denoise)
        
        if binarize_mode == "otsu":
            self.otsu_binarize_corrected(blur_ksize=5) # computes self.binary_mask + threshold
            print(f"Otsu threshold (corrected): {self.otsu_threshold:.1f}")
            self.plot_histogram_and_otsu()
        elif binarize_mode == "rowwise":
            self.rowwise_binarize_corrected(
                stat="mean",
                smooth_ksize=51,
                k=5.0,
                min_run=5,
                expand=2
            )
            print("Row-wise thresholding applied (no single global threshold).")
            self.plot_rowwise_threshold_debug()
        else:
            raise ValueError("binarize_mode must be 'otsu' or 'rowwise'")
        

        # Step 2: Split in half
        self.split_halves(use_corrected=True) #WL modified based on function
        mid_point = height // 2
        print(f"Split point: {mid_point} pixels")
        print(f"  Top half (control): rows 0-{mid_point}")
        print(f"  Bottom half (test): rows {mid_point}-{height}")
        # Now have the top half of the image and bottom half as separate images stored

        # Step 3: Estimate background from top 10% of image (should be white)
        # top_region = self.inverted_image[:int(0.1 * height), :]
        # self.background = np.mean(top_region)
        # WL replaced line:
        # top_region = self.corrected_image[:int(0.1 * height), :] # This takes the top 10% of the rows of the corrected image
        # self.background = np.mean(top_region) # Computes the median of them all -> estimate of the residual or remaining baseline/noise floor

        self.background, blank_mask = self.estimate_background_percentile_mask(
            which="inverted",      # use RAW inverted image
            bg_percentile=60.0     # try 50–70 depending on strictness
        )

        print(f"\nBackground (blank-mask percentile): {self.background:.2f}")


        print(f"\nBackground level: {self.background:.2f}")

        # Step 4: Find control line
        self.control_line_pos, self.control_intensity, _ = \
            self.find_darkest_line(self.top_half, "Control") # Compute the mean intensity of each row in top half. Pick the row with the maximum mean (brightest band)

        control_signal = self.control_intensity - self.background # Take the control and subtract that mean row intensity
        print(f"\nControl Line:")
        print(f"  Position: row {self.control_line_pos} (in top half)")
        print(f"  Intensity: {self.control_intensity:.2f}")
        print(f"  Signal above background: {control_signal:.2f}")

        # Step 5: Find test line
        self.test_line_pos, self.test_intensity, _ = \
            self.find_darkest_line(self.bottom_half, "Test")

        test_signal = self.test_intensity - self.background
        print(f"\nTest Line:")
        print(f"  Position: row {self.test_line_pos} (in bottom half)")
        print(f"  Intensity: {self.test_intensity:.2f}")
        print(f"  Signal above background: {test_signal:.2f}")

        # Step 6: Check if test line is real
        """
        adaptive_T = self.compute_adaptive_threshold(method="mad", k=5.0, bg_fraction=0.10)
        print(f"Adaptive threshold (row-mean units): {adaptive_T:.2f}")

        # Compare TEST signal above baseline against adaptive threshold above baseline
        # Since your signals are (row_mean - self.background), you can compare row_mean directly to T
        # OR compare signals to (T - background). We'll do signal-space for consistency:
        signal_threshold = adaptive_T - self.background
        print(f"Adaptive threshold (signal units): {signal_threshold:.2f}")
        """
        signal_threshold = min_test_threshold

        if test_signal < signal_threshold: # If the test line signal is less than the threshold we decided, call it a negative
            print(f"  ⚠️  Signal below threshold ({signal_threshold}) - NEGATIVE")
            self.is_negative = True
            self.test_intensity = self.background  # Set test_intensity to background value
            test_signal = 0.0
        else:
            print(f"  ✓ Signal above threshold - POSITIVE") # Signal is positive, there is a control line
            self.is_negative = False

        # Step 7: Calculate relative intensity
        control_signal = max(control_signal, 1.0)  # Avoid division by zero
        test_signal = max(test_signal, 0.0)

        self.relative_intensity = test_signal / control_signal # This takes relative intensity = test signal / control signal

        print(f"\nRelative Intensity:")
        print(f"  Test signal / Control signal = {test_signal:.2f} / {control_signal:.2f}")
        print(f"  = {self.relative_intensity:.4f}")

        print("=" * 70)
        print(f"RESULT: {'NEGATIVE' if self.is_negative else 'POSITIVE'}")
        print(f"Relative Intensity: {self.relative_intensity:.4f}")
        print("=" * 70)

        return {
            'control_intensity': self.control_intensity,
            'test_intensity': self.test_intensity,
            'background': self.background,
            'relative_intensity': self.relative_intensity,
            'is_negative': self.is_negative,
            'control_position': self.control_line_pos,
            'test_position': self.test_line_pos
        }

    def visualize(self, save_path=None):
        """Create simple 2-panel visualization: original and enhanced only"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

        height = self.inverted_image.shape[0]
        mid_point = height // 2
        skip = 3

        # Panel 1: Original image with split line
        ax1.imshow(cv2.cvtColor(self.original_image, cv2.COLOR_BGR2RGB))
        ax1.axhline(y=mid_point, color='yellow', linewidth=3, linestyle='--', label='Split line')
        ax1.set_title('Original Image', fontsize=14, fontweight='bold')
        ax1.legend(fontsize=11)
        ax1.axis('off')

        # Panel 2: Enhanced image with detected lines
        # ax2.imshow(self.inverted_image, cmap='hot')
        # WL: replacement
        ax2.imshow(self.corrected_image if self.corrected_image is not None else self.inverted_image, cmap='hot')
        ax2.axhline(y=mid_point, color='cyan', linewidth=2, linestyle='--', label='Split', alpha=0.7)



        # Show skipped regions
        ax2.axhspan(0, skip, alpha=0.3, color='gray', label='Skipped (edge)')
        ax2.axhspan(mid_point-skip, mid_point+skip, alpha=0.3, color='gray')
        ax2.axhspan(height-skip, height, alpha=0.3, color='gray')

        # Mark detected control line
        if self.control_line_pos is not None:
            # Control line position is relative to top half (which starts at skip)
            actual_control_pos = skip + self.control_line_pos
            ax2.axhline(y=actual_control_pos, color='cyan', linewidth=3, label='Control Line')

        # Mark detected test line
        if self.test_line_pos is not None:
            # Test line position is relative to bottom half (which starts at mid_point+skip)
            actual_test_pos = mid_point + skip + self.test_line_pos
            if not self.is_negative:
                ax2.axhline(y=actual_test_pos, color='lime', linewidth=3, label='Test Line')
            else:
                ax2.axhline(y=actual_test_pos, color='red', linewidth=3, linestyle=':', label='Test (below threshold)')

        ax2.set_title('Enhanced Image\n(Brighter = darker lines)', fontsize=14, fontweight='bold')
        ax2.legend(fontsize=10, loc='upper right')
        ax2.axis('off')


        # --- Put legends outside on the right ---
        # Make room on the right for legends
        fig.subplots_adjust(right=0.78)

        # Legends: anchor them outside each axes
        ax1.legend(fontsize=11, loc='upper left', bbox_to_anchor=(1.02, 1.0), borderaxespad=0.)
        ax2.legend(fontsize=10, loc='upper left', bbox_to_anchor=(1.02, 1.0), borderaxespad=0.)
        # ------------------------------------------


        # Overall title
        status = "NEGATIVE" if self.is_negative else "POSITIVE"
        fig.suptitle(f'{Path(self.image_path).name} - Relative Intensity: {self.relative_intensity:.4f} ({status})',
                    fontsize=16, fontweight='bold', y=0.98)

        # WL: Panel 3 -> print the inverted image, the estimated background, and the corrected image
        plt.figure(figsize=(12,4))
        plt.subplot(1,3,1); plt.title("Inverted"); plt.imshow(self.inverted_image, cmap="hot"); plt.axis("off")
        plt.subplot(1,3,2); plt.title("Estimated Background"); plt.imshow(self.background_image, cmap="hot"); plt.axis("off")
        plt.subplot(1,3,3); plt.title("Corrected (Inverted - Background)"); plt.imshow(self.corrected_image, cmap="hot"); plt.axis("off")

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Visualization saved: {save_path}")

        return fig




    def estimate_background_percentile_mask(
        self,
        which="inverted",          # use RAW inverted for quant
        bg_percentile=50.0,        # 50=median, 60–80 can be safer
        exclude_top=0.10,          # don't include very top edge if it has artifacts
        exclude_bottom=0.10,       # don't include very bottom edge (common gradient/edge)
        exclude_center_band=0.20,  # exclude a middle band where lines might exist
        edge_margin=0.05           # exclude left/right edges
    ):
        """
        Estimate background from a 'blank mask' using a percentile of selected pixels.

        Returns
        -------
        background_value : float
        mask : np.ndarray(bool)
        """

        # Pick image to estimate background on
        if which == "inverted":
            if self.inverted_image is None:
                raise ValueError("Run preprocess() first.")
            img = self.inverted_image.astype(np.float32)
        elif which == "gray":
            if self.gray_image is None:
                raise ValueError("Run preprocess() first.")
            img = self.gray_image.astype(np.float32)
        else:
            raise ValueError("which must be 'inverted' or 'gray'")

        h, w = img.shape
        mask = np.ones((h, w), dtype=bool)

        # Exclude top/bottom bands
        top_cut = int(exclude_top * h)
        bot_cut = int(exclude_bottom * h)
        if top_cut > 0:
            mask[:top_cut, :] = False
        if bot_cut > 0:
            mask[h - bot_cut:, :] = False

        # Exclude center band (where lines often live)
        center_half = int(0.5 * exclude_center_band * h)
        center = h // 2
        mask[max(0, center - center_half):min(h, center + center_half), :] = False

        # Exclude left/right edges
        edge = int(edge_margin * w)
        if edge > 0:
            mask[:, :edge] = False
            mask[:, w - edge:] = False

        # Pull masked pixels and compute percentile
        pixels = img[mask]
        if pixels.size == 0:
            raise ValueError("Blank mask removed all pixels; relax exclusions.")

        background_value = float(np.percentile(pixels, bg_percentile))
        return background_value, mask

    def plot_raw_row_stats_heatmap(self, stat="mean", tile_width=30):
      """
      Plot raw (un-background-subtracted) row statistics as a heatmap grid
      for top and bottom halves.

      Parameters
      ----------
      stat : str
          "mean" or "mode"
      tile_width : int
          How many columns to tile the 1D row stats into for visualization.
          Bigger => looks more like a strip.
      """

      if self.inverted_image is None:
          raise ValueError("Run preprocess() or analyze() first.")

      img = self.inverted_image
      height = img.shape[0]
      mid_point = height // 2
      skip = 3

      top_half = img[skip:mid_point-skip, :]
      bottom_half = img[mid_point+skip:height-skip, :]

      if stat == "mean":
          top_stats = np.mean(top_half, axis=1)
          bottom_stats = np.mean(bottom_half, axis=1)

      elif stat == "mode":
          def row_mode(arr):
              return np.array([np.bincount(row.astype(int)).argmax() for row in arr])
          top_stats = row_mode(top_half)
          bottom_stats = row_mode(bottom_half)

      else:
          raise ValueError("stat must be 'mean' or 'mode'")

      # Turn 1D vectors into 2D grids by tiling
      top_grid = np.tile(top_stats[:, None], (1, tile_width))
      bottom_grid = np.tile(bottom_stats[:, None], (1, tile_width))

      # fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 6))
      fig, (ax1, ax2) = plt.subplots(
          1, 2,
          figsize=(10, 6),
          constrained_layout=True  # ✅ Fix
      )

      im1 = ax1.imshow(top_grid, aspect="auto", cmap="hot")
      ax1.set_title(f"Top Half RAW Row {stat.capitalize()} Heatmap")
      ax1.set_xlabel("Tiled columns")
      ax1.set_ylabel("Row index")

      im2 = ax2.imshow(bottom_grid, aspect="auto", cmap="hot")
      ax2.set_title(f"Bottom Half RAW Row {stat.capitalize()} Heatmap")
      ax2.set_xlabel("Tiled columns")
      ax2.set_ylabel("Row index")

      # Shared colorbar
      cbar = fig.colorbar(im2, ax=[ax1, ax2], fraction=0.046, pad=0.04)
      cbar.set_label("Row statistic value")

      # plt.tight_layout()
      plt.show()

      return top_stats, bottom_stats