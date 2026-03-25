"""GIRO/DIDBase real-time ionospheric data client."""

import json
import urllib.request
import urllib.error
import urllib.parse
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any
import threading
import time


@dataclass
class Ionosonde:
    """Ionosonde station information."""

    station_id: str
    name: str
    latitude: float
    longitude: float
    ursi_code: str


@dataclass
class Ionogram:
    """Ionogram data from GIRO."""

    station_id: str
    timestamp: datetime
    foF2: Optional[float] = None  # MHz
    hmF2: Optional[float] = None  # km
    foF1: Optional[float] = None
    foE: Optional[float] = None
    hmE: Optional[float] = None
    fmin: Optional[float] = None
    muf_3000: Optional[float] = None
    confidence: float = 0.0  # ARTIST confidence score


class GIROClient:
    """Client for GIRO (Global Ionospheric Radio Observatory) data.

    Accesses real-time ionospheric data from DIDBase (Digital Ionogram
    DataBase) via the GIRO network.

    Data source: https://giro.uml.edu/
    """

    BASE_URL = "https://lgdc.uml.edu/common/DIDBGetValues"

    # List of major ionosonde stations
    STATIONS = {
        "WP937": Ionosonde("WP937", "Wallops Island", 37.9, -75.5, "WP937"),
        "BC840": Ionosonde("BC840", "Boulder", 40.0, -105.3, "BC840"),
        "EG931": Ionosonde("EG931", "Eglin AFB", 30.5, -86.5, "EG931"),
        "JR055": Ionosonde("JR055", "Jicamarca", -12.0, -76.9, "JR055"),
        "DB049": Ionosonde("DB049", "Dourbes", 50.1, 4.6, "DB049"),
        "RO041": Ionosonde("RO041", "Rome", 41.9, 12.5, "RO041"),
        "MH453": Ionosonde("MH453", "Millstone Hill", 42.6, -71.5, "MH453"),
        "AS00Q": Ionosonde("AS00Q", "Ascension Island", -7.9, -14.4, "AS00Q"),
    }

    def __init__(
        self,
        station_id: str = "WP937",
        update_interval_min: float = 15.0,
        auto_update: bool = False,
    ):
        """Initialize GIRO client.

        Args:
            station_id: Ionosonde station URSI code
            update_interval_min: Auto-update interval in minutes
            auto_update: Whether to automatically fetch new data
        """
        self._station_id = station_id
        self._update_interval = update_interval_min
        self._auto_update = auto_update

        self._current: Optional[Ionogram] = None
        self._history: List[Ionogram] = []
        self._callbacks = []

        self._update_thread: Optional[threading.Thread] = None
        self._running = False

        if auto_update:
            self.start_auto_update()

    @property
    def station(self) -> Optional[Ionosonde]:
        """Get current station info."""
        return self.STATIONS.get(self._station_id)

    @property
    def current(self) -> Optional[Ionogram]:
        """Get most recent ionogram data."""
        return self._current

    def set_station(self, station_id: str):
        """Change ionosonde station.

        Args:
            station_id: URSI code of station
        """
        if station_id in self.STATIONS:
            self._station_id = station_id
            self._current = None
            self._history.clear()
        else:
            print(f"Unknown station: {station_id}")

    def fetch_latest(self) -> Optional[Ionogram]:
        """Fetch latest ionogram data from GIRO.

        Returns:
            Ionogram data or None if unavailable
        """
        try:
            # Build query URL
            end_time = datetime.utcnow()
            start_time = end_time - timedelta(hours=1)

            params = {
                "ursiCode": self._station_id,
                "charName": "foF2,hmF2,foE,hmE,foF1,fmin,MUF(3000)F2",
                "DMUF": "3000",
                "fromDate": start_time.strftime("%Y-%m-%d %H:%M:%S"),
                "toDate": end_time.strftime("%Y-%m-%d %H:%M:%S"),
            }

            query = urllib.parse.urlencode(params)
            url = f"{self.BASE_URL}?{query}"

            # Fetch data
            req = urllib.request.Request(url)
            req.add_header("User-Agent", "HFPathSim/0.1")

            with urllib.request.urlopen(req, timeout=10) as response:
                data = response.read().decode("utf-8")

            # Parse response
            ionogram = self._parse_response(data)

            if ionogram:
                self._current = ionogram
                self._history.append(ionogram)

                # Notify callbacks
                for callback in self._callbacks:
                    callback(ionogram)

            return ionogram

        except urllib.error.URLError as e:
            print(f"GIRO fetch error: {e}")
            return None
        except Exception as e:
            print(f"GIRO parse error: {e}")
            return None

    def _parse_response(self, data: str) -> Optional[Ionogram]:
        """Parse GIRO response data.

        Args:
            data: Raw response text

        Returns:
            Parsed Ionogram or None
        """
        # GIRO returns space-separated values
        # Format varies, this handles common cases
        lines = data.strip().split("\n")

        if len(lines) < 2:
            return None

        # Find most recent valid entry
        for line in reversed(lines):
            parts = line.strip().split()
            if len(parts) < 3:
                continue

            try:
                # Parse timestamp (format varies)
                timestamp_str = f"{parts[0]} {parts[1]}"
                try:
                    timestamp = datetime.strptime(
                        timestamp_str, "%Y-%m-%d %H:%M:%S"
                    )
                except ValueError:
                    timestamp = datetime.utcnow()

                ionogram = Ionogram(
                    station_id=self._station_id,
                    timestamp=timestamp,
                )

                # Parse characteristic values
                for i, value in enumerate(parts[2:]):
                    try:
                        val = float(value)
                        if val < 0:
                            continue

                        # Map position to characteristic
                        if i == 0:
                            ionogram.foF2 = val
                        elif i == 1:
                            ionogram.hmF2 = val
                        elif i == 2:
                            ionogram.foE = val
                        elif i == 3:
                            ionogram.hmE = val
                        elif i == 4:
                            ionogram.foF1 = val
                        elif i == 5:
                            ionogram.fmin = val
                        elif i == 6:
                            ionogram.muf_3000 = val

                    except ValueError:
                        continue

                if ionogram.foF2 is not None:
                    return ionogram

            except Exception:
                continue

        return None

    def start_auto_update(self):
        """Start automatic data updates."""
        if self._running:
            return

        self._running = True
        self._update_thread = threading.Thread(
            target=self._update_loop, daemon=True
        )
        self._update_thread.start()

    def stop_auto_update(self):
        """Stop automatic data updates."""
        self._running = False
        if self._update_thread:
            self._update_thread.join(timeout=2.0)

    def _update_loop(self):
        """Background update loop."""
        while self._running:
            self.fetch_latest()
            time.sleep(self._update_interval * 60)

    def add_callback(self, callback):
        """Register callback for new data."""
        self._callbacks.append(callback)

    def remove_callback(self, callback):
        """Remove data callback."""
        if callback in self._callbacks:
            self._callbacks.remove(callback)

    def get_history(self, hours: float = 24) -> List[Ionogram]:
        """Get historical ionogram data.

        Args:
            hours: Number of hours of history

        Returns:
            List of Ionogram objects
        """
        cutoff = datetime.utcnow() - timedelta(hours=hours)
        return [i for i in self._history if i.timestamp > cutoff]

    @staticmethod
    def list_stations() -> Dict[str, Ionosonde]:
        """Get dictionary of available stations."""
        return GIROClient.STATIONS.copy()

    def to_vogler_params(
        self,
        frequency_mhz: float = 10.0,
        path_length_km: float = 1000.0,
        doppler_spread_hz: float = 1.0,
        delay_spread_ms: float = 2.0,
    ):
        """Convert current data to Vogler parameters."""
        from hfpathsim.core.parameters import VoglerParameters

        if self._current is None:
            return VoglerParameters()

        return VoglerParameters(
            foF2=self._current.foF2 or 7.0,
            hmF2=self._current.hmF2 or 300.0,
            foE=self._current.foE or 3.0,
            hmE=self._current.hmE or 110.0,
            frequency_mhz=frequency_mhz,
            path_length_km=path_length_km,
            doppler_spread_hz=doppler_spread_hz,
            delay_spread_ms=delay_spread_ms,
        )
