# Optical Test and Metrology

Last Updated: 2026-06-12

Research Status: Initial Framework

---

## Definition

Optical Test and Metrology has two separate industry lines.

Optical Communication Test measures optical signals, optical chips, optical modules and optical networks themselves.

Primary drivers:

- AI data center capex
- GPU cluster expansion
- Optical module shipments
- 800G / 1.6T / 3.2T transition
- CPO adoption
- Network deployment

Relevant public exposures:

- KEYS
- VIAV
- AEHR
- FORM

Semiconductor Optical Metrology uses optical methods to inspect wafers, process steps, chips and advanced packaging.

Primary drivers:

- Semiconductor capex
- Leading-edge nodes
- HBM
- Advanced packaging
- Inspection intensity

Relevant public exposures:

- KLAC
- ONTO
- NVMI
- CAMT

These two lines have different customers, capex cycles, products and competitive structures. They should not be placed in the same sector-strength bucket or described as the same cycle.

---

## Core Thesis

Optical testing is yield insurance.

Test Spending = Insurance Premium

Failure Probability = Probability of Loss

Package Scrap Cost = Insured Value

Core transmission:

Higher Speed
+
More Channels
+
Higher Integration
↓
Higher Failure Probability
+
Higher Failure Cost
↓
More Testing
+
Earlier Testing
+
Higher-Value Test Equipment

If each channel must pass independently, total package yield declines multiplicatively as channel count rises. This is a directionally important relationship, not a fixed assumption about industry yield levels.

---

## Why CPO Changes Testing

Pluggable Module:

Failure can often be isolated and replaced.

CPO:

- Optical engine is integrated close to expensive switching silicon.
- Late-stage failure becomes more expensive.
- Testing must move earlier.

Testing moves earlier through this path:

Wafer
↓
Bare Die
↓
Package
↓
Module / System

The economic change is not only higher optical volume. It is also the rising cost of failure after photonics, switching silicon and packaging are integrated.

---

## Demand Formula

Optical Test Demand =
Optical Unit Volume
×
Test Intensity
×
Test Coverage
×
Test Value per Station

Optical Unit Volume is driven by:

- GPU cluster size
- optical attach rate
- networking bandwidth
- data center capex

Test Intensity is driven by:

- speed
- channel count
- modulation complexity
- package integration

Test Coverage can be high during new product introduction:

- 100% test
- multiple temperatures
- multiple voltages
- repeated qualification

Test Coverage may normalize as products mature:

- sampling rate may decline
- test coverage may normalize

Test Value per Station can rise as standards move from:

800G
↓
1.6T
↓
3.2T

---

## Four Test Layers

### Layer 1 - Wafer-Level Optical Test

Tests:

- optical power
- spectrum
- dark current
- LIV
- wafer-level reliability
- Known Good Die

Equipment:

- optical probe station
- source-measure unit
- burn-in system
- probe card
- optical probe

Relevant companies:

- KEYS
- AEHR
- FORM

### Layer 2 - R&D Test Instruments

Tests:

- bit error rate
- eye diagram
- jitter
- TDECQ
- optical power
- wavelength

Equipment:

- BERT
- sampling oscilloscope
- real-time oscilloscope
- clock recovery
- optical spectrum analyzer

Relevant companies:

- KEYS
- VIAV
- Anritsu

### Layer 3 - Production Test

Tests:

- final module performance
- production yield
- throughput
- multi-temperature reliability

Relevant companies:

- KEYS
- VIAV

### Layer 4 - Network Test and Maintenance

Equipment:

- OTDR
- optical power meter
- network test platform

Relevant companies:

- VIAV

---

## Industry Relationships

| Source | Transmission Mechanism | Target | Public Exposure | Horizon | Confidence |
| --- | --- | --- | --- | --- | --- |
| GPU Cluster Growth | Higher Networking Bandwidth -> 800G / 1.6T Adoption | R&D and Production Test Demand | KEYS, VIAV | Near term | High |
| CPO Adoption | Higher Package Integration -> Higher Cost of Late Failure | Wafer-Level Optical and Electrical Test | KEYS, AEHR, FORM | Medium term | Medium |
| Higher Package Value | Stronger Known Good Die Requirement | Wafer-Level and Package-Level Burn-In | AEHR | Medium term | Medium |
| Wafer-Level Optical Test | Electrical and Optical Contact Requirement | Probe Card and Optical Probe Demand | FORM | Medium term | Medium |
| New Optical Standards | New Test Methodologies | New Hardware and Software Licenses | KEYS, VIAV | Near term | High |
| Optical Module Capacity Expansion | More Production Test Stations | Test Equipment Demand | KEYS, VIAV | Near term | Medium |
| CPO Adoption | Sub-Micron Optical Alignment | Photonics Packaging Equipment | Limited pure US-listed exposure | Medium term | Medium |

Pure US-listed exposure is limited for sub-micron optical alignment. Do not force a US ticker into this relationship without verified revenue exposure.

---

## Bottlenecks

### Production Test Capacity

This has already appeared in the near-term cycle.

Drivers:

- 1.6T ramp
- module capacity expansion
- full production testing

### Wafer-Level Optical and Electrical Test

This may have the largest long-term space.

Core issues:

- test methodology is not fully standardized
- limited production-ready solutions
- customer qualification is still evolving

### Sub-Micron Optical Alignment

Demand can be strong, but supply can be limited by:

- field engineers
- installation
- customer qualification
- delivery capacity

---

## Deepest Moats

The deepest moats are:

- Standards
- Process Data
- Customer Qualification
- Measurement Credibility

The deepest moats are not simply equipment manufacturing capacity.

Capacity shortages can create a temporary cycle.

A test methodology that becomes the industry reference can create a long-duration toll road.

---

## Timing

### Near Term

1.6T production ramp.

Primary observations:

- KEYS
- VIAV

### Medium Term

CPO qualification and initial production.

Primary observations:

- KEYS
- AEHR
- FORM

### Long Term

Wafer-level optical test standards.

Core question:

Which company or platform becomes the reference methodology?

---

## Company Map

| Ticker | Role | Exposure Type | Main Driver | Main Risk |
| --- | --- | --- | --- | --- |
| KEYS | Standards, R&D test, high-speed electrical/optical test, silicon photonics wafer test | Direct but diluted by diversified business | 1.6T / 3.2T standards and CPO qualification | Diversified exposure and capex timing |
| VIAV | Optical network test, lab test, production Ethernet test, field test | Direct optical/network-test exposure | Optical deployment and Ethernet speed upgrade | Telecom and data-center cycle mix |
| AEHR | Wafer-level burn-in, package-level burn-in, consumable contactors | Direct but small-cap and volatile | Known Good Die and reliability-test demand | Customer concentration and order volatility |
| FORM | Probe cards, probe stations, optical probing | Early-stage CPO optionality | Wafer-level optical and electrical contact | CPO exposure materiality and standards timing |

Separate research bucket:

- KLAC
- ONTO
- NVMI
- CAMT

These companies belong to Semiconductor Metrology & Inspection, not Optical Communication Test.

---

## Key Metrics

Industry Metrics:

- 800G shipment growth
- 1.6T qualification
- 1.6T shipment growth
- 3.2T roadmap
- CPO qualification milestones
- optical attach rate
- channels per package
- wafer-level optical-test adoption
- production test coverage
- module capex

KEYS Metrics:

- AI-related orders
- wireline test growth
- high-speed BERT demand
- oscilloscope demand
- silicon photonics commentary
- protocol software growth

VIAV Metrics:

- NSE organic growth
- data center test orders
- 1.6T and 3.2T product shipments
- integration margins
- cash conversion

AEHR Metrics:

- silicon photonics bookings
- book-to-bill
- new wafer-level customers
- package-level AI burn-in revenue
- consumable revenue

FORM Metrics:

- CPO-related revenue
- optical probe adoption
- system segment growth
- probe-card demand
- customer concentration

---

## Disconfirming Signals

- 1.6T deployment delays
- CPO adoption delays
- lower optical attach rate
- falling test coverage
- customers building internal test systems
- standard fragmentation delaying purchases
- optical revenue remaining immaterial
- architecture reducing burn-in needs
- module makers cutting capex

---

## Open Questions

1. Which wafer-level optical-test methodology becomes the reference standard?
2. Will testing be controlled by chip designers, foundries, OSATs or equipment vendors?
3. How much testing remains internal?
4. Does external-laser architecture reduce reliability-test intensity?
5. How much package redundancy offsets yield decline?
6. Which listed company first reports material CPO-related revenue?

---

## Source Discipline

This document is an industry thesis framework.

Company-specific revenue exposure, market share, valuation and product claims must be independently verified before being used in scoring or automated analysis.

Do not convert subjective rankings into fixed Investment OS ratings without verified sources.
