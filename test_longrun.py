import time


def main():
    for i in range(1, 101):
        value = (i * i + 3) / 7
        print(f"step={i:03d} value={value:.6f}", flush=True)
        time.sleep(1)


if __name__ == "__main__":
    main()
