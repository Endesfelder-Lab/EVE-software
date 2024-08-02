# Supplementary Methods

## Software environment
\* Python 3.9, specs of PCs, Windows/Linux versions

### *E. coli* sample preparation
*E. coli* MG1655 rpoC:mEos3.2-A69T (CamR) was grown o/n from -70°C glycerol stocks in M9 minimal medium (Merck) at 37°C under constant agitation. This culture was subcultured 1:100 in EZRDM (EZ rich defined medium; Teknova) complemented with 1:100 20% glucose, and grown for 24h, after which it was subcultured the same. This culture was grown for 3 hours, and the cells were directly fixed by adding 150 uL 37% glycerol (pre-warmed to 37°C) to 1350 uL culture and gentle agitation, and this fixation was continued for 20 minutes at 37°C. The cells were washed twice with 500 uL PBS at 37°C by centrifugation at 9000 g, and finally concentrated in 200 uL PBS. This sample was placed on an 8-chamber Ibidi slide, which was beforehand prepared as follows: the slide was cleaned by 15 minutes incubation with 1M KOH, washed twice with distilled water, incubated with poly-L-lysine (PLL) for 20 minutes, washed twice with distilled water. After incubation of the sample on the slide, the sample was washed twice with PBS before imaging.
### Microscopy details – DNA-PAINT origami and E. coli imaging
A custom laser-based fluorescence microscopy setup was used for imaging. 561 nm laser light (gem 561 1000, Laser Quantum) was modulated by an AOTF (G&H; AOMO 3080-125 & AODR 1080AF-AINA-1.0 HCR) and directed via a reflective collimator (RC04FC-P01, Thorlabs) into a custom fibre (70x70 µm multimode square core fibre, NA 0.22, FC/PC connectors, CeramOptec). 405 nm laser light (LBX-405-1200-HPE-PPA, RPMC) was co-aligned and also directed into the same fibre. The fibre output was collimated via a Multimode Collimator (F950FC-A 350-700nm, Thorlabs), and the beam was expanded via a set of lenses (LB1471-A-ML and LA4725-A, Thorlabs), and cleaned up via a ZET405/488/561/640 filter (QuadLineLaserClean-Up, AHF analysentechnik). This beam was then focused via an achromatic lens (AC254-400-A-ML, Thorlabs) and a dichroic mirror (ZT405/488/561rpc, Chroma, Bellows Falls, VT, USA) embedded in a commercial inverted microscope body (Nikon Eclipse Ti-E, Nikon, Tokyo, Japan) equipped with a focus stabilization system (Perfect focus system) on the back-focal plane of a 60x Apochromat TIRF 1.49 NA objective (Nikon). Emission from the sample passed via the objective and dichroic mirror through a filter (ZET405/488/561m-TRF, Chroma) and a 4f system via two convex achromatic lenses (AC508-100-A, Thorlabs), and optionally passed through an emission filter (ET610/75m, Chroma). The light was then directed either towards a Prime BSI sCMOS camera (Teledyne Photometrics, Tucson, AZ, USA; 107 nm pixel size), or towards an event-based sensor (Metavision Gen4.1-HD EVK, Prophesee). The microscope, camera, and peripherals were controlled via MicroManager 2.0, and laser triggering was controlled via a TriggerScope 4 (Advanced Research Consulting, Newcastle, CA, USA).

For *E. coli* rpoC imaging, the sample was illuminated with ~2 kW/cm<sup>2</sup> 561 nm laser, while the 405nm laser was increased manually from 0 to ~8 W/cm<sup>2</sup> to have a low and steady photoactivation rate, until no new signal appeared. For the Nile Red imaging, the buffer was exchanged for PBS containing 12.5 nM Nile Red, and the sample was illuminated with ~2 kW/cm<sup>2</sup> 561 nm laser and the data was recorded for 3 minutes.

For E. coli Nile Red analysis, the frame based finding method (Detection threshold = 3.0, Exclusion radius = 4.0, Min. Radius = 1.25, Max. Radius = 4.0, Frame time (ms) = 100.0 and Candidate radius = 4.0) was used for positive and negative events separately.  

For DNA-PAINT nanoruler (80RG, Gattaquant) imaging, the sample was illuminated with ~2 kW/cm<sup>2</sup> 561 nm laser in TIRF mode by moving the focused spot to the side of the back-focal plane of the objective, and ~20 minutes of data was recorded.

*Analysis method nanoruler - Laura

\* Analysis methods – Joel - rpoC

# A-tubulin
\* Biological – Manon/Clement

\* France setup– Manon/Clement

\* Analysis methods - Laura

