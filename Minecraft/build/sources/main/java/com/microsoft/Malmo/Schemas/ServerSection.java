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
 *       &lt;sequence>
 *         &lt;element ref="{http://ProjectMalmo.microsoft.com}ServerInitialConditions" minOccurs="0"/>
 *         &lt;element ref="{http://ProjectMalmo.microsoft.com}ServerHandlers"/>
 *       &lt;/sequence>
 *     &lt;/restriction>
 *   &lt;/complexContent>
 * &lt;/complexType>
 * </pre>
 * 
 * 
 */
@XmlAccessorType(XmlAccessType.FIELD)
@XmlType(name = "", propOrder = {
    "serverInitialConditions",
    "serverHandlers"
})
@XmlRootElement(name = "ServerSection")
public class ServerSection {

    @XmlElement(name = "ServerInitialConditions")
    protected ServerInitialConditions serverInitialConditions;
    @XmlElement(name = "ServerHandlers", required = true)
    protected ServerHandlers serverHandlers;

    /**
     * Gets the value of the serverInitialConditions property.
     * 
     * @return
     *     possible object is
     *     {@link ServerInitialConditions }
     *     
     */
    public ServerInitialConditions getServerInitialConditions() {
        return serverInitialConditions;
    }

    /**
     * Sets the value of the serverInitialConditions property.
     * 
     * @param value
     *     allowed object is
     *     {@link ServerInitialConditions }
     *     
     */
    public void setServerInitialConditions(ServerInitialConditions value) {
        this.serverInitialConditions = value;
    }

    /**
     * Gets the value of the serverHandlers property.
     * 
     * @return
     *     possible object is
     *     {@link ServerHandlers }
     *     
     */
    public ServerHandlers getServerHandlers() {
        return serverHandlers;
    }

    /**
     * Sets the value of the serverHandlers property.
     * 
     * @param value
     *     allowed object is
     *     {@link ServerHandlers }
     *     
     */
    public void setServerHandlers(ServerHandlers value) {
        this.serverHandlers = value;
    }

}
