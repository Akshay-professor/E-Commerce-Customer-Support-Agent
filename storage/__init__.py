# Marks `storage` as a regular package (not a namespace package) so submodule
# imports like `from storage import orders_store` resolve reliably across
# environments, including Streamlit Cloud's multipage launcher.
