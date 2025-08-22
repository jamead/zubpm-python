import matplotlib.pyplot as plt
import numpy as np
import time

def initialize_plot():
    """
    Initializes the plot and returns the figure and axes objects.
    """
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(10, 8))

    # Create plot lines for each subplot
    ax1 = axes[0, 0]
    line_a, = ax1.plot([], [], 'b-o')
    ax1.set_title('ChA')
    ax1.set_ylabel('ADU')
    ax1.grid()

    ax2 = axes[0, 1]
    line_b, = ax2.plot([], [], 'r-o')
    ax2.set_title('ChB')
    ax2.set_ylabel('ADU')
    ax2.grid()

    ax3 = axes[1, 0]
    line_c, = ax3.plot([], [], 'g-o')
    ax3.set_title('ChC')
    ax3.set_ylabel('ADU')
    ax3.set_xlabel('Sample Number')
    ax3.grid()

    ax4 = axes[1, 1]
    line_d, = ax4.plot([], [], 'm-o')
    ax4.set_title('ChD')
    ax4.set_ylabel('ADU')
    ax4.set_xlabel('Sample Number')
    ax4.grid()

    titlestr = "ADC Data: 4 Channels"
    fig.suptitle(titlestr, fontsize=14)
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)  # Adjust to make room for the title

    # Show the plot immediately (this is very important to trigger the display)
    plt.show()

    return fig, line_a, line_b, line_c, line_d, ax1, ax2, ax3, ax4

def plot_adc_data(a, b, c, d, line_a, line_b, line_c, line_d):
    """
    Update the plot with new ADC data.
    """
    line_a.set_xdata(np.arange(len(a)))
    line_a.set_ydata(a)

    line_b.set_xdata(np.arange(len(b)))
    line_b.set_ydata(b)

    line_c.set_xdata(np.arange(len(c)))
    line_c.set_ydata(c)

    line_d.set_xdata(np.arange(len(d)))
    line_d.set_ydata(d)

    # Redraw the plot to update it
    plt.draw()
    plt.pause(0.1)  # Give time to update the plot

def main():
    # Initialize the plot and get the line objects
    fig, line_a, line_b, line_c, line_d, ax1, ax2, ax3, ax4 = initialize_plot()

    # Loop to continuously update the plot
    for _ in range(10):  # Set a fixed number of iterations for testing
        # Simulate new data for each channel (replace with real-time data here)
        a = np.random.random(10) * 1000  # Example random data for Channel A
        b = np.random.random(10) * 1000  # Example random data for Channel B
        c = np.random.random(10) * 1000  # Example random data for Channel C
        d = np.random.random(10) * 1000  # Example random data for Channel D

        # Update the plot with new data
        plot_adc_data(a, b, c, d, line_a, line_b, line_c, line_d)

        # Sleep for a bit before the next update (simulate real-time data fetching)
        time.sleep(1)

# Run the main function
if __name__ == "__main__":
    main()

