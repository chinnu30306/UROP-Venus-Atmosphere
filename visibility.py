from astroquery.jplhorizons import Horizons

obj = Horizons(
    id="499",
    location="599",
    epochs={"start": "2064-06-17 00:00",
            "stop": "2064-06-18 00:00",
            "step": "1h"}
)

eph = obj.ephemerides()

for row in eph:
    print(
        row["datetime_str"],
        "RA =", row["RA"],
        "Dec =", row["DEC"],
        "Solar elongation =", row["elong"]
    )