import json
import sys


def main():
    res = []
    for filename in sys.argv[1:]:
        print(f"file: {filename}")
        with open(f"{filename}.txt") as file:
            texts = set(file.readlines())
            for text in texts:
                text = text.replace("…", " ").strip()
                if text:
                    res.append(dict(text=text, meaning=int(filename)-1))
    with open(f"res.json", "w") as resfile:
        resfile.write( json.dumps(res, ensure_ascii=False, indent=4))

if __name__ == "__main__":
    main()