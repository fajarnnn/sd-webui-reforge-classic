import threading
import time
from modules import shared

def ensure_uni_pc_order():
    # Loop sampai shared.opts sudah siap
    while getattr(shared, "opts", None) is None:
        print("⏳ Menunggu shared.opts siap...")
        time.sleep(1)

    if not hasattr(shared.opts, "uni_pc_order"):
        print("⚙️ Menambahkan opsi uni_pc_order ke shared.opts...")
        setattr(shared.opts, "uni_pc_order", "default")
    else:
        print("✅ Opsi uni_pc_order sudah ada.")

# Jalankan di thread terpisah supaya gak ganggu startup
threading.Thread(target=ensure_uni_pc_order, daemon=True).start()
