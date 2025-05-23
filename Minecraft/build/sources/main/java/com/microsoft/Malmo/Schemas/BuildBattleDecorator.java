//
// This file was generated by the JavaTM Architecture for XML Binding(JAXB) Reference Implementation, v2.2.4 
// See <a href="http://java.sun.com/xml/jaxb">http://java.sun.com/xml/jaxb</a> 
// Any modifications to this file will be lost upon recompilation of the source schema. 
// Generated on: 2025.05.21 at 01:07:37 PM PDT 
//


package com.microsoft.Malmo.Schemas;

import javax.xml.bind.annotation.XmlAccessType;
import javax.xml.bind.annotation.XmlAccessorType;
import javax.xml.bind.annotation.XmlElement;
import javax.xml.bind.annotation.XmlRootElement;
import javax.xml.bind.annotation.XmlType;


/**
 * <p>Java class for anonymous complex type.
 * 
 * <p>The following schema fragment specifies the expected content contained within this class.
 * 
 * <pre>
 * &lt;complexType>
 *   &lt;complexContent>
 *     &lt;restriction base="{http://www.w3.org/2001/XMLSchema}anyType">
 *       &lt;all>
 *         &lt;element name="GoalStructureBounds" type="{http://ProjectMalmo.microsoft.com}UnnamedGridDefinition"/>
 *         &lt;element name="PlayerStructureBounds" type="{http://ProjectMalmo.microsoft.com}UnnamedGridDefinition"/>
 *         &lt;element name="BlockTypeOnCorrectPlacement" type="{http://ProjectMalmo.microsoft.com}DrawBlockBasedObjectType" minOccurs="0"/>
 *         &lt;element name="BlockTypeOnIncorrectPlacement" type="{http://ProjectMalmo.microsoft.com}DrawBlockBasedObjectType" minOccurs="0"/>
 *       &lt;/all>
 *     &lt;/restriction>
 *   &lt;/complexContent>
 * &lt;/complexType>
 * </pre>
 * 
 * 
 */
@XmlAccessorType(XmlAccessType.FIELD)
@XmlType(name = "", propOrder = {

})
@XmlRootElement(name = "BuildBattleDecorator")
public class BuildBattleDecorator {

    @XmlElement(name = "GoalStructureBounds", required = true)
    protected UnnamedGridDefinition goalStructureBounds;
    @XmlElement(name = "PlayerStructureBounds", required = true)
    protected UnnamedGridDefinition playerStructureBounds;
    @XmlElement(name = "BlockTypeOnCorrectPlacement")
    protected DrawBlockBasedObjectType blockTypeOnCorrectPlacement;
    @XmlElement(name = "BlockTypeOnIncorrectPlacement")
    protected DrawBlockBasedObjectType blockTypeOnIncorrectPlacement;

    /**
     * Gets the value of the goalStructureBounds property.
     * 
     * @return
     *     possible object is
     *     {@link UnnamedGridDefinition }
     *     
     */
    public UnnamedGridDefinition getGoalStructureBounds() {
        return goalStructureBounds;
    }

    /**
     * Sets the value of the goalStructureBounds property.
     * 
     * @param value
     *     allowed object is
     *     {@link UnnamedGridDefinition }
     *     
     */
    public void setGoalStructureBounds(UnnamedGridDefinition value) {
        this.goalStructureBounds = value;
    }

    /**
     * Gets the value of the playerStructureBounds property.
     * 
     * @return
     *     possible object is
     *     {@link UnnamedGridDefinition }
     *     
     */
    public UnnamedGridDefinition getPlayerStructureBounds() {
        return playerStructureBounds;
    }

    /**
     * Sets the value of the playerStructureBounds property.
     * 
     * @param value
     *     allowed object is
     *     {@link UnnamedGridDefinition }
     *     
     */
    public void setPlayerStructureBounds(UnnamedGridDefinition value) {
        this.playerStructureBounds = value;
    }

    /**
     * Gets the value of the blockTypeOnCorrectPlacement property.
     * 
     * @return
     *     possible object is
     *     {@link DrawBlockBasedObjectType }
     *     
     */
    public DrawBlockBasedObjectType getBlockTypeOnCorrectPlacement() {
        return blockTypeOnCorrectPlacement;
    }

    /**
     * Sets the value of the blockTypeOnCorrectPlacement property.
     * 
     * @param value
     *     allowed object is
     *     {@link DrawBlockBasedObjectType }
     *     
     */
    public void setBlockTypeOnCorrectPlacement(DrawBlockBasedObjectType value) {
        this.blockTypeOnCorrectPlacement = value;
    }

    /**
     * Gets the value of the blockTypeOnIncorrectPlacement property.
     * 
     * @return
     *     possible object is
     *     {@link DrawBlockBasedObjectType }
     *     
     */
    public DrawBlockBasedObjectType getBlockTypeOnIncorrectPlacement() {
        return blockTypeOnIncorrectPlacement;
    }

    /**
     * Sets the value of the blockTypeOnIncorrectPlacement property.
     * 
     * @param value
     *     allowed object is
     *     {@link DrawBlockBasedObjectType }
     *     
     */
    public void setBlockTypeOnIncorrectPlacement(DrawBlockBasedObjectType value) {
        this.blockTypeOnIncorrectPlacement = value;
    }

}
