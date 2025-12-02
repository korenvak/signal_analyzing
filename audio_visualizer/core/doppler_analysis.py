"""
Core logic for Doppler effect analysis and curve fitting.
"""
import numpy as np
from scipy.optimize import curve_fit
from dataclasses import dataclass
from typing import Tuple, Optional, List, Dict
import logging

logger = logging.getLogger(__name__)

@dataclass
class DopplerResult:
    """Results of a Doppler analysis."""
    velocity: float        # Velocity in m/s
    cpa_distance: float    # Closest Point of Approach distance in meters
    t_cpa: float          # Time of Closest Point of Approach
    f0: float             # Rest frequency in Hz
    rmse: float           # Root Mean Square Error of the fit
    direction: str        # "Approaching" or "Receding" (based on time relative to t_cpa)
    
    @property
    def velocity_kmh(self) -> float:
        return self.velocity * 3.6

class DopplerAnalyzer:
    """Analyzer for fitting Doppler curves to time-frequency data."""
    
    def __init__(self, speed_of_sound: float = 343.0):
        """
        Args:
            speed_of_sound: Speed of sound in the medium (m/s). Default 343.0 (Air).
        """
        self.c = speed_of_sound

    def doppler_function(self, t, f0, v, x_cpa, t_cpa):
        """
        Theoretical Doppler shift formula for a source moving in a straight line
        past a stationary observer.
        
        f(t) = f0 * (c / (c - v_radial))
        
        Where v_radial is the component of velocity towards the observer:
        v_radial = -v * (v * (t - t_cpa)) / sqrt(x_cpa^2 + v^2 * (t - t_cpa)^2)
        
        Note: The sign of v_radial depends on convention. 
        Here: t < t_cpa => approaching => higher freq.
        """
        # Avoid division by zero
        denom_sqrt = np.sqrt(x_cpa**2 + v**2 * (t - t_cpa)**2)
        denom_sqrt = np.maximum(denom_sqrt, 1e-6) # Epsilon
        
        # Radial velocity component (negative when approaching, positive when receding if using standard convention)
        # But simpler formulation:
        # cos(theta) = (v * (t - t_cpa)) / distance
        # v_radial = v * cos(theta)
        
        # When approaching (t < t_cpa), v_radial should be negative relative to observer?
        # Let's use the standard scalar formula:
        # f = f0 * c / (c + v_radial)
        # If source is approaching, v_radial is negative.
        
        # Geometry:
        # Source position at time t relative to CPA: P(t) = v * (t - t_cpa) along track.
        # Distance D(t) = sqrt(x_cpa^2 + P(t)^2)
        # Radial velocity is dD/dt.
        # D(t)^2 = x_cpa^2 + v^2(t-t_cpa)^2
        # 2 D dD/dt = 2 v^2 (t-t_cpa)
        # v_radial = dD/dt = (v^2 * (t - t_cpa)) / D(t)
        
        v_radial = (v**2 * (t - t_cpa)) / denom_sqrt
        
        # Observed frequency
        # f_obs = f_source * c / (c + v_source_radial) 
        # (assuming receiver static)
        
        return f0 * self.c / (self.c + v_radial)

    def fit_curve(self, 
                  times: np.ndarray, 
                  freqs: np.ndarray, 
                  initial_f0: Optional[float] = None) -> Optional[DopplerResult]:
        """
        Fit a Doppler curve to the provided points.
        
        Args:
            times: Array of time stamps (seconds)
            freqs: Array of frequency values (Hz)
            initial_f0: Optional guess for f0. If None, estimated from data.
            
        Returns:
            DopplerResult object or None if fit failed.
        """
        if len(times) < 4:
            logger.warning("Not enough points for Doppler fit (min 4)")
            return None
            
        # Normalize times to avoid huge numbers in optimization
        t_mean = np.mean(times)
        t_centered = times - t_mean
        
        # Initial Guesses
        if initial_f0 is None:
            # Use MEDIAN frequency as the rest frequency (f0)
            # This is the actual emitted frequency, not affected by Doppler
            # The median is more robust than mean for skewed distributions
            p0_f0 = np.median(freqs)
            logger.info(f"Using median frequency as f0: {p0_f0:.1f} Hz")
        else:
            p0_f0 = initial_f0
            
        # Estimate t_cpa (where frequency changes most rapidly, or just middle of time)
        # For standard pass-by, freq drops. CPA is where freq is roughly f0.
        p0_t_cpa = times[np.argmin(np.abs(freqs - p0_f0))] - t_mean
        
        # Estimate v and x_cpa roughly
        # This is hard without more context, but we can try reasonable bounds
        p0_v = 20.0 # 20 m/s ~ 72 km/h
        p0_x_cpa = 50.0 # 50 meters
        
        p0 = [p0_f0, p0_v, p0_x_cpa, p0_t_cpa]
        
        # Bounds: f0 > 0, v > 0, x_cpa > 0
        bounds = (
            [0, 0.1, 0.1, -np.inf], # Lower
            [np.inf, 340.0, np.inf, np.inf] # Upper (v < speed of sound)
        )
        
        try:
            popt, pcov = curve_fit(
                self.doppler_function, 
                t_centered, 
                freqs, 
                p0=p0, 
                bounds=bounds,
                maxfev=2000
            )
            
            # Unpack results
            fit_f0, fit_v, fit_x_cpa, fit_t_cpa_centered = popt
            
            # Correct t_cpa back to absolute time
            fit_t_cpa = fit_t_cpa_centered + t_mean
            
            # Calculate RMSE
            residuals = freqs - self.doppler_function(t_centered, *popt)
            rmse = np.sqrt(np.mean(residuals**2))
            
            return DopplerResult(
                velocity=fit_v,
                cpa_distance=fit_x_cpa,
                t_cpa=fit_t_cpa,
                f0=fit_f0,
                rmse=rmse,
                direction="Approaching" if t_mean < fit_t_cpa else "Receding"
            )
            
        except RuntimeError as e:
            logger.error(f"Doppler fit failed: {e}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error in Doppler fit: {e}")
            return None

    def generate_curve(self, result: DopplerResult, times: np.ndarray) -> np.ndarray:
        """Generate the theoretical frequency curve for a given time range."""
        # We need to pass t - t_mean if we were using centered times, 
        # but here we implemented the function to take raw t but we need to handle the offset carefully.
        # Actually, the doppler_function takes 't' and 't_cpa'. 
        # So we can just pass the result.t_cpa directly.
        
        return self.doppler_function(
            times, 
            result.f0, 
            result.velocity, 
            result.cpa_distance, 
            result.t_cpa
        )

