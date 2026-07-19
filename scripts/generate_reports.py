from house_price_predictor.reporting import generate_reports

if __name__ == "__main__":
    for path in generate_reports():
        print(path)
