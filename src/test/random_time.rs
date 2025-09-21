use chrono::{DateTime, Local, NaiveDate, TimeZone as _, Utc};
use rand::Rng;

pub fn random_time_between(start: DateTime<Utc>, end: DateTime<Utc>) -> DateTime<Utc> {
    let duration = end.signed_duration_since(start);
    let mut rng = rand::rng();
    let random_duration = rng.random_range(0..duration.num_seconds());
    start + chrono::Duration::seconds(random_duration)
}

pub fn random_date() -> NaiveDate {
    let start = Utc.with_ymd_and_hms(2021, 1, 1, 0, 0, 0).unwrap();
    let end = Local::now().with_timezone(&Utc);
    random_time_between(start, end).date_naive()
}
#[cfg(test)]
mod tests {
    use super::*;
    use chrono::Duration;
    #[test]
    fn test_random_time_between() {
        let start = Utc::now();
        let end = start + Duration::hours(1);
        let random_time = random_time_between(start, end);
        println!("Random time: {}", random_time);
        assert!(random_time >= start && random_time <= end);
    }
}
