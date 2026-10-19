# -*- coding: utf-8 -*-
"""
Opentrons OT-2 Automated Synthesis Protocol for Auto-Catalyst 2.0
Batch: 15.0g Composite Catalyst (USY + ZSM-5)
Target Liquid Dispensing:
  - USY Slurry: 20.29 mL
  - ZSM-5 Slurry: 9.71 mL
  - Alumina Sol: 25.0 mL
  - Kaolin Clay: 20.94 mL
  - RE Promoter: 0.3 mL
"""

metadata = {
    'protocolName': 'Auto-Catalyst 2.0: FCC Dual-Zeolite Composite Slurry Prep',
    'author': 'Antigravity Autonomous Lab System',
    'description': 'Automated dispensing and homogenization of dual-zeolite FCC catalyst precursors',
    'apiLevel': '2.14'
}

def run(protocol):
    # Labware setup
    # Slot 1: Reservoir containing reagent stock solutions (15mL Falcon tubes)
    reservoir = protocol.load_labware('opentrons_6_tuberack_falcon_50ml', '1')
    # Slot 2: Mixing/reaction vessel (temperature-controlled or shaker module)
    reaction_vessel = protocol.load_labware('opentrons_6_tuberack_falcon_50ml', '2')
    # Slot 3: Tip rack
    tiprack_1000 = protocol.load_labware('opentrons_96_tiprack_1000ul', '3')

    # Pipette setup (1000 uL single channel)
    p1000 = protocol.load_instrument('p1000_single_gen2', 'right', tip_racks=[tiprack_1000])

    # Reagent stock tube positions
    usy_stock = reservoir.wells_by_name()['A1']
    zsm5_stock = reservoir.wells_by_name()['A2']
    alumina_stock = reservoir.wells_by_name()['A3']
    kaolin_stock = reservoir.wells_by_name()['B1']
    re_stock = reservoir.wells_by_name()['B2']

    # Target synthesis reactor vessel
    target_reactor = reaction_vessel.wells_by_name()['A1']

    # Step 1: Dispense Alumina Sol Matrix (Base peptized binder)
    protocol.comment("Step 1: Dispensing Alumina Sol Matrix...")
    p1000.pick_up_tip()
    alumina_ul = 25000
    while alumina_ul > 0:
        transfer_vol = min(alumina_ul, 1000)
        p1000.aspirate(transfer_vol, alumina_stock.bottom(2))
        p1000.dispense(transfer_vol, target_reactor.top(-5))
        alumina_ul -= transfer_vol
    p1000.drop_tip()

    # Step 2: Dispense Kaolin Clay Suspension
    protocol.comment("Step 2: Dispensing Kaolin Clay Suspension...")
    p1000.pick_up_tip()
    kaolin_ul = 20940
    while kaolin_ul > 0:
        transfer_vol = min(kaolin_ul, 1000)
        p1000.aspirate(transfer_vol, kaolin_stock.bottom(2))
        p1000.dispense(transfer_vol, target_reactor.top(-5))
        kaolin_ul -= transfer_vol
    p1000.drop_tip()

    # Step 3: Dispense USY Zeolite Slurry
    protocol.comment("Step 3: Dispensing USY Zeolite Slurry...")
    p1000.pick_up_tip()
    p1000.mix(3, 800, usy_stock.bottom(5))  # Re-suspend zeolite before transfer
    usy_ul = 20290
    while usy_ul > 0:
        transfer_vol = min(usy_ul, 1000)
        p1000.aspirate(transfer_vol, usy_stock.bottom(2))
        p1000.dispense(transfer_vol, target_reactor.top(-5))
        usy_ul -= transfer_vol
    p1000.drop_tip()

    # Step 4: Dispense ZSM-5 Additive Slurry
    protocol.comment("Step 4: Dispensing ZSM-5 Additive Slurry...")
    p1000.pick_up_tip()
    p1000.mix(3, 800, zsm5_stock.bottom(5))
    zsm5_ul = 9710
    while zsm5_ul > 0:
        transfer_vol = min(zsm5_ul, 1000)
        p1000.aspirate(transfer_vol, zsm5_stock.bottom(2))
        p1000.dispense(transfer_vol, target_reactor.top(-5))
        zsm5_ul -= transfer_vol
    p1000.drop_tip()

    # Step 5: Dispense Rare-Earth Promoter
    re_ul = 300
    if re_ul > 0:
        protocol.comment("Step 5: Dispensing Rare-Earth Promoter...")
        p1000.pick_up_tip()
        while re_ul > 0:
            transfer_vol = min(re_ul, 1000)
            p1000.aspirate(transfer_vol, re_stock.bottom(2))
            p1000.dispense(transfer_vol, target_reactor.top(-5))
            re_ul -= transfer_vol
        p1000.drop_tip()

    # Final Homogenization mixing
    protocol.comment("Step 6: Performing Final Pre-Gel Homogenization...")
    p1000.pick_up_tip()
    p1000.mix(5, 900, target_reactor.bottom(10))
    p1000.drop_tip()

    protocol.comment("Automated composite slurry preparation complete. Transfer to aging and spray-drying.")
