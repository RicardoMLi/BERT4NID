from scapy.compat import raw
from scapy.layers.dns import DNS
from scapy.layers.inet import IP, TCP, UDP
from scapy.layers.l2 import Ether, ARP
from scapy.packet import Padding


def should_omit_packet(packet):

    # SYN, ACK or FIN flags set to 1 and no payload
    if TCP in packet and (packet.flags & 0x13):
        # not payload or contains only padding
        layers = packet[TCP].payload.layers()
        if not layers or (Padding in layers and len(layers) == 1):
            return True

    if UDP in packet and not packet[UDP].payload:
        return True

    # DNS segment
    if DNS in packet or ARP in packet:
        return True

    return False


def remove_ether_header(packet):
    if Ether in packet:
        return packet[Ether].payload

    return packet


def mask_mac(packet):
    if Ether in packet:
        packet.src = '00:00:00:00:00:00'
        packet.dst = '00:00:00:00:00:00'

    return packet


def mask_ip(packet):
    if IP in packet:
        packet[IP].src = "0.0.0.0"
        packet[IP].dst = "0.0.0.0"

    return packet


def mask_udp(packet):
    if UDP in packet:
        packet[UDP].sport = 0
        packet[UDP].dport = 0

    return packet


def mask_tcp(packet):
    if TCP in packet:
        packet[TCP].sport = 0
        packet[TCP].dport = 0

    return packet


def pad_udp(packet):
    if UDP in packet:
        # get layers after udp
        layer_after = packet[UDP].payload.copy()

        # build a padding layer
        pad = Padding()
        pad.load = "\x00" * 12

        layer_before = packet.copy()
        layer_before[UDP].remove_payload()
        packet = layer_before / pad / layer_after

        return packet

    return packet


def crop_and_pad(packet, max_length=1024) -> bytes:
    packet_bytes = bytearray(raw(packet))
    origin_len = len(packet_bytes)

    if origin_len < max_length:
        packet_bytes.extend(b'\x00' * (max_length - origin_len))
    else:
        packet_bytes = packet_bytes[:max_length]

    return bytes(packet_bytes)


def transform_packet(packet):
    if IP not in packet:
        return None

    packet = remove_ether_header(packet)
    packet = mask_ip(packet)
    packet = mask_udp(packet)
    packet = mask_tcp(packet)

    return packet
