# Data source: UCI Bike Sharing Dataset

## Official source

- Dataset: [Bike Sharing](https://archive.ics.uci.edu/dataset/275/bike+sharing+dataset)
- Repository: UCI Machine Learning Repository
- DOI: [10.24432/C5W894](https://doi.org/10.24432/C5W894)
- Dataset page reports 17,389 instances and 13 features, with no missing values.

## License

The UCI dataset page identifies the dataset license as **Creative Commons Attribution 4.0 International (CC BY 4.0)**. The license permits sharing and adaptation for any purpose provided appropriate credit is given. Review the [CC BY 4.0 legal code](https://creativecommons.org/licenses/by/4.0/) and retain attribution when redistributing adapted data.

## Bibliographic citation

Fanaee-T, H. (2013). *Bike Sharing* [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C5W894.

Related introductory paper listed by UCI:

Fanaee-T, H., & Gama, J. (2013). Event labeling combining ensemble detectors and background knowledge. *Progress in Artificial Intelligence*.

## Dataset description

The dataset contains hourly and daily rental-bike counts from the Capital Bikeshare system in Washington, D.C., for 2011–2012, together with calendar, seasonal, and weather information. The downloadable files listed by UCI are `hour.csv`, `day.csv`, and `Readme.txt`.

The hourly file includes the date and hour, season, year, month, holiday and weekday indicators, weather situation, normalized temperature, perceived temperature, humidity and wind speed, counts of casual and registered users, and total rentals (`cnt`). In the published schema, `cnt` is the sum of `casual` and `registered`. The daily file does not include `hr`.

## Limitations and interpretation

- The records cover one bike-sharing system in one city and only two years (2011–2012); results may not generalize to other locations, systems, or current demand.
- The source describes aggregate rental counts and associated context, not individual trips or station-level demand. It cannot support claims about a particular station or rider.
- Weather fields are normalized (rather than raw measurements), and several variables are encoded categorically. Refer to the source's variable descriptions before interpreting values.
- `cnt` is constructed from the casual and registered user counts. Those component columns must not be used as predictors when predicting `cnt`, because that would directly reveal the target.
- Forecasting evaluation must respect time order. Randomly splitting rows could leak future patterns into training; a future modeling stage should define its forecast horizon and temporal validation design.
- UCI reports no missing values, but project-specific schema, timestamp continuity, duplicates, and data integrity still need to be checked after acquisition.
- The dataset is historical and its target reflects the original system's measurement and operating context. It does not represent live availability or current ridership.

## Planned use in this project

The planned initial target is hourly total demand (`cnt`) from `hour.csv`. Downloading, validating, exploratory analysis, feature design, and forecasting are future work. No analysis or predictive capability is claimed as implemented at this stage.
