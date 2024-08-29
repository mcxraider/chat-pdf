export default function Input() {
  return (
    <div
      style={{
        padding: "1rem",
        backgroundColor: "#ffffff",
        borderTop: "1px solid #ddd",
      }}
    >
      <input
        type="text"
        placeholder="Type a message..."
        style={{
          width: "100%",
          padding: "10px",
          borderRadius: "5px",
          border: "1px solid #ddd",
        }}
        disabled // This input does nothing for now
      />
    </div>
  );
}
