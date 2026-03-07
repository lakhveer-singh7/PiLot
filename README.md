# PiLot — Pipeline Learning on Tiny Devices

On-device CNN training for nRF52840 microcontrollers (256 KB RAM, 1 MB Flash).  
Two implementations live side-by-side:

| Project | Description | Quick Start |
|---------|-------------|-------------|
| [PiLot_Centralized](PiLot_Centralized/) | Single-device training baseline | `cd PiLot_Centralized && ./run.sh` |
| [PiLot_Distributed](PiLot_Distributed/) | 7-device pipelined training | `cd PiLot_Distributed && ./run.sh` |

See each project's **README.md** for architecture details, control-flow diagrams, and configuration instructions.
