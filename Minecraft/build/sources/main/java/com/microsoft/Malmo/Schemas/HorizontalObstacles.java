//
// This file was generated by the JavaTM Architecture for XML Binding(JAXB) Reference Implementation, v2.2.4 
// See <a href="http://java.sun.com/xml/jaxb">http://java.sun.com/xml/jaxb</a> 
// Any modifications to this file will be lost upon recompilation of the source schema. 
// Generated on: 2025.05.21 at 01:07:37 PM PDT 
//


package com.microsoft.Malmo.Schemas;

import javax.xml.bind.annotation.XmlAccessType;
import javax.xml.bind.annotation.XmlAccessorType;
import javax.xml.bind.annotation.XmlType;


/**
 * <p>Java class for HorizontalObstacles complex type.
 * 
 * <p>The following schema fragment specifies the expected content contained within this class.
 * 
 * <pre>
 * &lt;complexType name="HorizontalObstacles">
 *   &lt;complexContent>
 *     &lt;restriction base="{http://www.w3.org/2001/XMLSchema}anyType">
 *       &lt;all>
 *         &lt;element name="gap" type="{http://ProjectMalmo.microsoft.com}NonNegative"/>
 *         &lt;element name="bridge" type="{http://ProjectMalmo.microsoft.com}NonNegative"/>
 *         &lt;element name="door" type="{http://ProjectMalmo.microsoft.com}NonNegative"/>
 *         &lt;element name="puzzle" type="{http://ProjectMalmo.microsoft.com}NonNegative"/>
 *         &lt;element name="jump" type="{http://ProjectMalmo.microsoft.com}NonNegative"/>
 *       &lt;/all>
 *     &lt;/restriction>
 *   &lt;/complexContent>
 * &lt;/complexType>
 * </pre>
 * 
 * 
 */
@XmlAccessorType(XmlAccessType.FIELD)
@XmlType(name = "HorizontalObstacles", propOrder = {

})
public class HorizontalObstacles {

    protected int gap;
    protected int bridge;
    protected int door;
    protected int puzzle;
    protected int jump;

    /**
     * Gets the value of the gap property.
     * 
     */
    public int getGap() {
        return gap;
    }

    /**
     * Sets the value of the gap property.
     * 
     */
    public void setGap(int value) {
        this.gap = value;
    }

    /**
     * Gets the value of the bridge property.
     * 
     */
    public int getBridge() {
        return bridge;
    }

    /**
     * Sets the value of the bridge property.
     * 
     */
    public void setBridge(int value) {
        this.bridge = value;
    }

    /**
     * Gets the value of the door property.
     * 
     */
    public int getDoor() {
        return door;
    }

    /**
     * Sets the value of the door property.
     * 
     */
    public void setDoor(int value) {
        this.door = value;
    }

    /**
     * Gets the value of the puzzle property.
     * 
     */
    public int getPuzzle() {
        return puzzle;
    }

    /**
     * Sets the value of the puzzle property.
     * 
     */
    public void setPuzzle(int value) {
        this.puzzle = value;
    }

    /**
     * Gets the value of the jump property.
     * 
     */
    public int getJump() {
        return jump;
    }

    /**
     * Sets the value of the jump property.
     * 
     */
    public void setJump(int value) {
        this.jump = value;
    }

}
