import pyopencl as cl

platforms = cl.get_platforms()
for p in platforms:
    print(f"Platform: {p.name}")
    for d in p.get_devices():
        print(f"  Device: {d.name} - {cl.device_type.to_string(d.type)}")
