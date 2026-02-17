from image_processing import *


# image_path = 'LFAIMAGES/'
image_path = '50_fold_manual_1.jpeg'

# if __name__ == "__main__":
#   analyzer = SimpleLFAAnalyzer("50_fold_manual_1.jpeg")
#   analyzer.preprocess()  # enough for raw inverted image
#   analyzer.plot_raw_row_stats_heatmap(stat="mean", tile_width=40)
  
  
  
if __name__ == "__main__":
    # analyzer = SimpleLFAAnalyzer("50_fold_manual_1.jpeg")
    # analyzer = SimpleLFAAnalyzer("75_fold_manual_1.jpeg")
    # analyzer = SimpleLFAAnalyzer("image7-75fold.jpeg")
    # analyzer = SimpleLFAAnalyzer("image9-75fold2.jpeg")
    analyzer = SimpleLFAAnalyzer(image_path)

    # Run full analysis (includes background subtraction + split_halves(use_corrected=True))
    results = analyzer.analyze(min_test_threshold=1.0, bg='morph_ellips_mod', k=51, normalize=False, denoise=False)

    # Optional debugging outputs
    # analyzer.plot_intensity_histogram()
    # analyzer.plot_intensity_heatmap()

    # Visualize detected lines
    # analyzer.visualize()
    # plt.show()

    # Print results dict at the end
    print("\nResults:", results)



# if __name__ == "__main__":
#     analyzer = SimpleLFAAnalyzer("50_fold_manual_1.jpeg")
#     results = analyzer.analyze()
#     analyzer.visualize()
#     plt.show()