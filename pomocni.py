import sounddevice as sd

print(sd.query_devices())

device_id = 1  # ovde stavi broj Remote Audio mikrofona
print(sd.query_devices(device_id))