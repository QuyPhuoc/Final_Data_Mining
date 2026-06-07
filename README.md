# Hướng dẫn cài đặt và sử dụng UV

## UV là gì?

`uv` là một công cụ quản lý package và môi trường Python cực nhanh được phát triển bởi :contentReference[oaicite:0]{index=0}.  
Nó có thể thay thế cho:

- `pip`
- `venv`
- `pip-tools`
- `poetry` (một phần)

---

# 1. Cài đặt UV
## macOS và Linux
Use curl to download the script and execute it with sh:
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```
If your system doesn't have curl, you can use wget:
```bash
wget -qO- https://astral.sh/uv/install.sh | sh
```
Request a specific version by including it in the URL:
```bash
curl -LsSf https://astral.sh/uv/0.11.13/install.sh | sh
```

## Window
Use irm to download the script and execute it with iex:
```bash
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```
Changing the execution policy allows running a script from the internet.

Request a specific version by including it in the URL:
```bash
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/0.11.13/install.ps1 | iex"
```
### How to use
Help menus
The --help flag can be used to view the help menu for a command, e.g., for uv:
```bash
uv --help
```
To view the help menu for a specific command, e.g., for uv init:
```bash
uv init --help
```
When using the --help flag, uv displays a condensed help menu. To view a longer help menu for a command, use uv help:
```bash
uv help
```
To view the long help menu for a specific command, e.g., for uv init:
```bash
uv help init
```
When using the long help menu, uv will attempt to use less or more to "page" the output so it is not all displayed at once. To exit the pager, press q.
## Displaying verbose output
The -v flag can be used to display verbose output for a command, e.g., for uv sync:
```bash
uv sync -v
```
The -v flag can be repeated to increase verbosity, e.g.:
```bash
uv sync -vv
```
Often, the verbose output will include additional information about why uv is behaving in a certain way.
## Creating a new project
You can create a new Python project using the uv init command:
```bash
uv init hello-world
cd hello-world
```
## Create virtual environments
```bash
uv venv
```
## Managing dependencies
### Co the add nhieu thu vien cung luc
```bash
uv add <ten_thu_vien> <ten_thu_vien> <ten_thu_vien> .... (vi du: uv add pandas numpy opencv ...)
```
